//
// TSP Branch-and-Bound paralelo (MPI). Rank 0 = master (bolsa de trabajo),
// ranks 1..N-1 = workers (computo pesado). Comunicacion de tNodos via MPI_Pack.
//
// Compilar: mpic++ -O2 tsppar.cpp libtsp.cc -o tsppar
//

#include "libtsp.h"
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ---- Tags ----
#define TAG_WORK      10  // master -> worker : nodos de trabajo (PACKED)
#define TAG_NEED      20  // worker -> master : "estoy ocioso, dame trabajo" (INT)
#define TAG_TOPW2M    30  // worker -> master : nueva mejor solucion (PACKED)
#define TAG_TOPM2W    40  // master -> worker : update del top-10 (PACKED)
#define TAG_STEALREQ  50  // master -> worker : peticion de trabajo (INT)
#define TAG_STEALRESP 60  // worker -> master : respuesta a peticion (PACKED)
#define TAG_TERM      70  // master -> worker : terminacion (INT)
#define TAG_TERMACK   80  // worker -> master : confirmacion de terminacion (INT)

#define TOPN          10
#define MIN_SPLIT      4  // tam. minimo de pila para poder ceder trabajo

// ---------------------------------------------------------------------------
//  Helpers de (des)empaquetado de un tNodo (inspirados en tspparv2)
// ---------------------------------------------------------------------------
static int node_pack_size() {
    int a, b, c, d;
    MPI_Pack_size(1, MPI_LONG, MPI_COMM_WORLD, &a); // id
    MPI_Pack_size(2, MPI_INT, MPI_COMM_WORLD, &b); // ci, orig_excl
    MPI_Pack_size(NCIUDADES, MPI_INT, MPI_COMM_WORLD, &c); // incl
    MPI_Pack_size(NCIUDADES - 2, MPI_INT, MPI_COMM_WORLD, &d); // dest_excl
    return a + b + c + d;
}

static void pack_node(char *buf, int cap, int &pos, const tNodo *n) {
    MPI_Pack((void *) &n->id, 1, MPI_LONG, buf, cap, &pos, MPI_COMM_WORLD);
    MPI_Pack((void *) &n->ci, 1, MPI_INT, buf, cap, &pos, MPI_COMM_WORLD);
    MPI_Pack((void *) &n->orig_excl, 1, MPI_INT, buf, cap, &pos, MPI_COMM_WORLD);
    MPI_Pack((void *) n->incl, NCIUDADES, MPI_INT, buf, cap, &pos, MPI_COMM_WORLD);
    MPI_Pack((void *) n->dest_excl, NCIUDADES - 2, MPI_INT, buf, cap, &pos, MPI_COMM_WORLD);
}

static void unpack_node(char *buf, int cap, int &pos, tNodo *n) {
    MPI_Unpack(buf, cap, &pos, &n->id, 1, MPI_LONG, MPI_COMM_WORLD);
    MPI_Unpack(buf, cap, &pos, &n->ci, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Unpack(buf, cap, &pos, &n->orig_excl, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Unpack(buf, cap, &pos, n->incl, NCIUDADES, MPI_INT, MPI_COMM_WORLD);
    MPI_Unpack(buf, cap, &pos, n->dest_excl, NCIUDADES - 2, MPI_INT, MPI_COMM_WORLD);
}

// ---------------------------------------------------------------------------
//  Top-10 (orden descendente por ci: top[0] = peor de los mejores = cota).
//  Copia profunda con CopiaNodo para evitar aliasing de los punteros internos.
// ---------------------------------------------------------------------------
static bool update_top(tNodo *best, tNodo top[TOPN]) {
    int pos = -1;
    for (int i = 0; i < TOPN; i++) {
        if (top[i].ci > best->ci) pos = i;
        else break;
    }
    if (pos < 0) return false;
    for (int i = 0; i < pos; i++) CopiaNodo(&top[i + 1], &top[i], false);
    CopiaNodo(best, &top[pos], false);
    return true;
}

static void init_top(tNodo top[TOPN]) {
    for (int i = 0; i < TOPN; i++) {
        InicNodo(&top[i]);
        top[i].ci = INFINITO;
    }
}

// Volcado periodico del estado del top-10 y del peor coste (CS)
static void print_top(tNodo top[TOPN], long long it) {
    printf("\n===== [master] iteracion %lld =====\n", it);
    printf("CS (peor coste del top-10) = %d\n", top[0].ci);
    for (int i = TOPN - 1; i >= 0; i--) {
        // del mejor al peor
        printf("  #%d  ", TOPN - 1 - i);
        if (top[i].ci >= INFINITO) printf("(vacio)\n");
        else {
            EscribeNodo(&top[i]);
            printf("\n");
        }
    }
    fflush(stdout);
}

// ---------------------------------------------------------------------------
//  Generacion de trabajo inicial: al menos 2 nodos por worker en la bolsa.
// ---------------------------------------------------------------------------
static void generar_trabajo_inicial(int num_workers, int **tsp, tPila *pool,
                                    tNodo top[TOPN]) {
    tNodo root, left, right;
    InicNodo(&root);
    InicNodo(&left);
    InicNodo(&right);

    int **tmp = reservarMatrizCuadrada(NCIUDADES);
    for (int i = 0; i < (int) NCIUDADES; i++)
        for (int j = 0; j < (int) NCIUDADES; j++) tmp[i][j] = tsp[i][j];
    int ci0 = 0;
    Reduce(tmp, &ci0);
    liberarMatriz(tmp);
    root.ci = ci0;

    while (PilaTamanio(pool) < 2 * num_workers && !PilaLlena(pool)) {
        Ramifica(&root, &left, &right, tsp);
        if (Solucion(&left)) update_top(&left, top);
        else PilaPush(pool, &left);
        if (Solucion(&right)) update_top(&right, top);
        else PilaPush(pool, &right);
        if (PilaVacia(pool)) break;
        PilaPop(pool, &root);
    }
}

// ===========================================================================
//  MASTER
// ===========================================================================
static void run_master(int N, int **tsp) {
    const int nw = N - 1;
    const int nsz = node_pack_size();
    int isz;
    MPI_Pack_size(1, MPI_INT, MPI_COMM_WORLD, &isz);
    const int wcap = isz + 2 * nsz; // 1 int (count) + 2 nodos
    const int scap = 2 * isz + MAXPILA * nsz; // status + count + nodos

    tPila pool;
    PilaInic(&pool);
    tNodo top[TOPN];
    init_top(top);

    generar_trabajo_inicial(nw, tsp, &pool, top);

    // Buffers y requests por worker (indices 1..N-1)
    char **bwork = new char *[N];
    char **btop = new char *[N];
    char **brnw = new char *[N]; // recv NEED
    char **brtop = new char *[N]; // recv TOP worker->master
    char **brstl = new char *[N]; // recv STEAL response
    MPI_Request *rqwork = new MPI_Request[N];
    MPI_Request *rqtop = new MPI_Request[N];
    MPI_Request *rqneed = new MPI_Request[N];
    MPI_Request *rqtw = new MPI_Request[N];
    int *peticion = new int[N]; // worker pidio trabajo (ocioso)
    int *needbuf = new int[N];

    for (int w = 1; w < N; w++) {
        bwork[w] = new char[wcap];
        btop[w] = new char[nsz];
        brnw[w] = new char[isz];
        brtop[w] = new char[nsz];
        brstl[w] = new char[scap];
        rqwork[w] = MPI_REQUEST_NULL;
        rqtop[w] = MPI_REQUEST_NULL;
        peticion[w] = 0;
        MPI_Irecv(&needbuf[w], 1, MPI_INT, w, TAG_NEED, MPI_COMM_WORLD, &rqneed[w]);
        MPI_Irecv(brtop[w], nsz, MPI_PACKED, w, TAG_TOPW2M, MPI_COMM_WORLD, &rqtw[w]);
    }

    // Reparto inicial: 2 nodos a cada worker
    for (int w = 1; w < N; w++) {
        int cnt = 0, pos = 0;
        tNodo a, b;
        InicNodo(&a);
        InicNodo(&b);
        if (!PilaVacia(&pool)) {
            PilaPop(&pool, &a);
            cnt++;
        }
        if (!PilaVacia(&pool)) {
            PilaPop(&pool, &b);
            cnt++;
        }
        MPI_Pack(&cnt, 1, MPI_INT, bwork[w], wcap, &pos, MPI_COMM_WORLD);
        if (cnt >= 1) pack_node(bwork[w], wcap, pos, &a);
        if (cnt >= 2) pack_node(bwork[w], wcap, pos, &b);
        MPI_Isend(bwork[w], pos, MPI_PACKED, w, TAG_WORK, MPI_COMM_WORLD, &rqwork[w]);
    }

    long long iter = 0;
    bool terminar = false;
    while (!terminar) {
        // Volcado del top-10 y de CS cada millon de iteraciones
        if (++iter % 1000000 == 0) print_top(top, iter);

        // 1) Updates de top-10 que llegan de los workers
        for (int w = 1; w < N; w++) {
            int flag = 0;
            MPI_Test(&rqtw[w], &flag, MPI_STATUS_IGNORE);
            if (flag) {
                tNodo s;
                InicNodo(&s);
                int pos = 0;
                unpack_node(brtop[w], nsz, pos, &s);
                if (update_top(&s, top)) {
                    // mejoro: reenviar a todos
                    for (int k = 1; k < N; k++) {
                        if (rqtop[k] != MPI_REQUEST_NULL)
                            MPI_Wait(&rqtop[k], MPI_STATUS_IGNORE);
                        int p2 = 0;
                        pack_node(btop[k], nsz, p2, &s);
                        MPI_Isend(btop[k], p2, MPI_PACKED, k, TAG_TOPM2W,
                                  MPI_COMM_WORLD, &rqtop[k]);
                    }
                }
                MPI_Irecv(brtop[w], nsz, MPI_PACKED, w, TAG_TOPW2M,
                          MPI_COMM_WORLD, &rqtw[w]);
            }
        }

        // 2) Peticiones de trabajo de workers ociosos
        for (int w = 1; w < N; w++) {
            int flag = 0;
            MPI_Test(&rqneed[w], &flag, MPI_STATUS_IGNORE);
            if (flag) {
                peticion[w] = 1;
                MPI_Irecv(&needbuf[w], 1, MPI_INT, w, TAG_NEED,
                          MPI_COMM_WORLD, &rqneed[w]);
            }
        }

        // 3) Servir trabajo desde la bolsa a los workers que lo pidieron
        for (int w = 1; w < N && !PilaVacia(&pool); w++) {
            if (!peticion[w]) continue;
            if (rqwork[w] != MPI_REQUEST_NULL)
                MPI_Wait(&rqwork[w], MPI_STATUS_IGNORE);
            tNodo a, b;
            InicNodo(&a);
            InicNodo(&b);
            int cnt = 0, pos = 0;
            if (!PilaVacia(&pool)) {
                PilaPop(&pool, &a);
                cnt++;
            }
            if (!PilaVacia(&pool)) {
                PilaPop(&pool, &b);
                cnt++;
            }
            MPI_Pack(&cnt, 1, MPI_INT, bwork[w], wcap, &pos, MPI_COMM_WORLD);
            if (cnt >= 1) pack_node(bwork[w], wcap, pos, &a);
            if (cnt >= 2) pack_node(bwork[w], wcap, pos, &b);
            MPI_Isend(bwork[w], pos, MPI_PACKED, w, TAG_WORK,
                      MPI_COMM_WORLD, &rqwork[w]);
            peticion[w] = 0;
        }

        // 4) Bolsa vacia: si hay workers ociosos, robar trabajo / detectar fin
        if (PilaVacia(&pool)) {
            int ociosos = 0;
            for (int w = 1; w < N; w++) ociosos += peticion[w];
            if (ociosos > 0) {
                int dummy = 0;
                for (int w = 1; w < N; w++)
                    MPI_Send(&dummy, 1, MPI_INT, w, TAG_STEALREQ, MPI_COMM_WORLD);

                MPI_Request *rr = new MPI_Request[N];
                for (int w = 1; w < N; w++)
                    MPI_Irecv(brstl[w], scap, MPI_PACKED, w, TAG_STEALRESP,
                              MPI_COMM_WORLD, &rr[w]);
                for (int w = 1; w < N; w++)
                    MPI_Wait(&rr[w], MPI_STATUS_IGNORE);
                delete[] rr;

                int todos_fin = 1;
                for (int w = 1; w < N; w++) {
                    int pos = 0, status = 0, cnt = 0;
                    MPI_Unpack(brstl[w], scap, &pos, &status, 1, MPI_INT, MPI_COMM_WORLD);
                    if (status == 2) {
                        todos_fin = 0;
                        MPI_Unpack(brstl[w], scap, &pos, &cnt, 1, MPI_INT, MPI_COMM_WORLD);
                        for (int k = 0; k < cnt && !PilaLlena(&pool); k++) {
                            tNodo n;
                            InicNodo(&n);
                            unpack_node(brstl[w], scap, pos, &n);
                            PilaPush(&pool, &n);
                        }
                    } else if (status == 1) {
                        todos_fin = 0;
                    }
                }
                if (todos_fin) terminar = true;
            }
        }
    }

    // ----------------- Terminacion ordenada -----------------
    // 1) Esperar los envios pendientes del master
    for (int w = 1; w < N; w++) {
        if (rqwork[w] != MPI_REQUEST_NULL) MPI_Wait(&rqwork[w], MPI_STATUS_IGNORE);
        if (rqtop[w] != MPI_REQUEST_NULL) MPI_Wait(&rqtop[w], MPI_STATUS_IGNORE);
    }
    // 2) Avisar de terminacion a todos los workers
    for (int w = 1; w < N; w++) {
        int dummy = 0;
        MPI_Send(&dummy, 1, MPI_INT, w, TAG_TERM, MPI_COMM_WORLD);
    }
    // 3) Drenar NEED/TOPW2M aun en vuelo y recoger el ACK de cada worker
    int *ackbuf = new int[N];
    MPI_Request *rqack = new MPI_Request[N];
    for (int w = 1; w < N; w++)
        MPI_Irecv(&ackbuf[w], 1, MPI_INT, w, TAG_TERMACK,
                  MPI_COMM_WORLD, &rqack[w]);

    int acks = 0;
    while (acks < N - 1) {
        for (int w = 1; w < N; w++) {
            int f;
            f = 0;
            MPI_Test(&rqneed[w], &f, MPI_STATUS_IGNORE);
            if (f)
                MPI_Irecv(&needbuf[w], 1, MPI_INT, w, TAG_NEED,
                          MPI_COMM_WORLD, &rqneed[w]);
            f = 0;
            MPI_Test(&rqtw[w], &f, MPI_STATUS_IGNORE);
            if (f)
                MPI_Irecv(brtop[w], nsz, MPI_PACKED, w, TAG_TOPW2M,
                          MPI_COMM_WORLD, &rqtw[w]);
            if (rqack[w] != MPI_REQUEST_NULL) {
                f = 0;
                MPI_Test(&rqack[w], &f, MPI_STATUS_IGNORE);
                if (f) acks++;
            }
        }
    }
    // 4) Drenado final: ningun NEED/TOPW2M debe quedar sin emparejar
    int restante = 1;
    while (restante) {
        restante = 0;
        for (int w = 1; w < N; w++) {
            int p, f;
            MPI_Status st;
            MPI_Iprobe(w, TAG_NEED, MPI_COMM_WORLD, &p, &st);
            if (p) {
                restante = 1;
                f = 0;
                MPI_Test(&rqneed[w], &f, MPI_STATUS_IGNORE);
                if (f)
                    MPI_Irecv(&needbuf[w], 1, MPI_INT, w, TAG_NEED,
                              MPI_COMM_WORLD, &rqneed[w]);
            }
            MPI_Iprobe(w, TAG_TOPW2M, MPI_COMM_WORLD, &p, &st);
            if (p) {
                restante = 1;
                f = 0;
                MPI_Test(&rqtw[w], &f, MPI_STATUS_IGNORE);
                if (f)
                    MPI_Irecv(brtop[w], nsz, MPI_PACKED, w, TAG_TOPW2M,
                              MPI_COMM_WORLD, &rqtw[w]);
            }
        }
    }
    delete[] ackbuf;
    delete[] rqack;

    // Volcado final + escritura de la mejor solucion (top[TOPN-1])
    print_top(top, iter);
    EscribeSolucion(&top[TOPN - 1], 0.0);

    // 5) Cancelar los Irecv persistentes (ya sin mensajes pendientes) y liberar
    for (int w = 1; w < N; w++) {
        if (rqneed[w] != MPI_REQUEST_NULL) {
            MPI_Cancel(&rqneed[w]);
            MPI_Request_free(&rqneed[w]);
        }
        if (rqtw[w] != MPI_REQUEST_NULL) {
            MPI_Cancel(&rqtw[w]);
            MPI_Request_free(&rqtw[w]);
        }
        delete[] bwork[w];
        delete[] btop[w];
        delete[] brnw[w];
        delete[] brtop[w];
        delete[] brstl[w];
    }
    delete[] bwork;
    delete[] btop;
    delete[] brnw;
    delete[] brtop;
    delete[] brstl;
    delete[] rqwork;
    delete[] rqtop;
    delete[] rqneed;
    delete[] rqtw;
    delete[] peticion;
    delete[] needbuf;
}

// ===========================================================================
//  WORKER
// ===========================================================================
static void run_worker(int rank, int **tsp) {
    const int nsz = node_pack_size();
    int isz;
    MPI_Pack_size(1, MPI_INT, MPI_COMM_WORLD, &isz);
    const int wcap = isz + 2 * nsz;
    const int scap = 2 * isz + MAXPILA * nsz;

    tPila stack;
    PilaInic(&stack);
    tNodo top[TOPN];
    init_top(top);

    char *bwork = new char[wcap];
    char *btopm = new char[nsz];
    char *btopw = new char[nsz];
    char *bstl = new char[scap];
    int steal_dummy = 0, term_dummy = 0, need_id = rank;

    MPI_Request rqwork = MPI_REQUEST_NULL, rqtopm = MPI_REQUEST_NULL;
    MPI_Request rqstl = MPI_REQUEST_NULL, rqterm = MPI_REQUEST_NULL;
    MPI_Request rqneed = MPI_REQUEST_NULL, rqtopw = MPI_REQUEST_NULL;

    MPI_Irecv(bwork, wcap, MPI_PACKED, 0, TAG_WORK, MPI_COMM_WORLD, &rqwork);
    MPI_Irecv(btopm, nsz, MPI_PACKED, 0, TAG_TOPM2W, MPI_COMM_WORLD, &rqtopm);
    MPI_Irecv(&steal_dummy, 1, MPI_INT, 0, TAG_STEALREQ, MPI_COMM_WORLD, &rqstl);
    MPI_Irecv(&term_dummy, 1, MPI_INT, 0, TAG_TERM, MPI_COMM_WORLD, &rqterm);

    bool need_sent = false;
    bool running = true;

    while (running) {
        int flag;

        // a) Terminacion
        flag = 0;
        MPI_Test(&rqterm, &flag, MPI_STATUS_IGNORE);
        if (flag) {
            running = false;
            break;
        }

        // b) Update del top global desde el master
        flag = 0;
        MPI_Test(&rqtopm, &flag, MPI_STATUS_IGNORE);
        if (flag) {
            tNodo s;
            InicNodo(&s);
            int pos = 0;
            unpack_node(btopm, nsz, pos, &s);
            update_top(&s, top);
            MPI_Irecv(btopm, nsz, MPI_PACKED, 0, TAG_TOPM2W,
                      MPI_COMM_WORLD, &rqtopm);
        }

        // c) Trabajo enviado por el master
        flag = 0;
        MPI_Test(&rqwork, &flag, MPI_STATUS_IGNORE);
        if (flag) {
            int pos = 0, cnt = 0;
            MPI_Unpack(bwork, wcap, &pos, &cnt, 1, MPI_INT, MPI_COMM_WORLD);
            for (int k = 0; k < cnt && !PilaLlena(&stack); k++) {
                tNodo n;
                InicNodo(&n);
                unpack_node(bwork, wcap, pos, &n);
                PilaPush(&stack, &n);
            }
            need_sent = false;
            MPI_Irecv(bwork, wcap, MPI_PACKED, 0, TAG_WORK,
                      MPI_COMM_WORLD, &rqwork);
        }

        // d) Peticion de trabajo del master (work-stealing inverso)
        flag = 0;
        MPI_Test(&rqstl, &flag, MPI_STATUS_IGNORE);
        if (flag) {
            int pos = 0;
            if (PilaTamanio(&stack) >= MIN_SPLIT) {
                tPila otra;
                PilaInic(&otra);
                PilaDivide(&stack, &otra);
                int status = 2, cnt = PilaTamanio(&otra);
                MPI_Pack(&status, 1, MPI_INT, bstl, scap, &pos, MPI_COMM_WORLD);
                MPI_Pack(&cnt, 1, MPI_INT, bstl, scap, &pos, MPI_COMM_WORLD);
                tNodo n;
                InicNodo(&n);
                while (!PilaVacia(&otra)) {
                    PilaPop(&otra, &n);
                    pack_node(bstl, scap, pos, &n);
                }
                MPI_Send(bstl, pos, MPI_PACKED, 0, TAG_STEALRESP, MPI_COMM_WORLD);
            } else {
                int status = PilaVacia(&stack) ? 0 : 1; // 0=ocioso 1=ocupado sin reparto
                MPI_Pack(&status, 1, MPI_INT, bstl, scap, &pos, MPI_COMM_WORLD);
                MPI_Send(bstl, pos, MPI_PACKED, 0, TAG_STEALRESP, MPI_COMM_WORLD);
            }
            MPI_Irecv(&steal_dummy, 1, MPI_INT, 0, TAG_STEALREQ,
                      MPI_COMM_WORLD, &rqstl);
        }

        // e) Computo pesado
        if (PilaVacia(&stack)) {
            if (!need_sent) {
                // avisar que estoy ocioso
                if (rqneed != MPI_REQUEST_NULL)
                    MPI_Wait(&rqneed, MPI_STATUS_IGNORE);
                MPI_Isend(&need_id, 1, MPI_INT, 0, TAG_NEED,
                          MPI_COMM_WORLD, &rqneed);
                need_sent = true;
            }
            continue;
        }

        tNodo nodo;
        InicNodo(&nodo);
        PilaPop(&stack, &nodo);
        if (nodo.ci >= top[0].ci) continue; // poda por cota

        tNodo left, right;
        InicNodo(&left);
        InicNodo(&right);
        Ramifica(&nodo, &left, &right, tsp);

        tNodo *hijos[2] = {&left, &right};
        for (int h = 0; h < 2; h++) {
            tNodo *c = hijos[h];
            if (c->ci >= top[0].ci) continue;
            if (Solucion(c)) {
                if (update_top(c, top)) {
                    // entra en el top -> avisar
                    if (rqtopw != MPI_REQUEST_NULL)
                        MPI_Wait(&rqtopw, MPI_STATUS_IGNORE);
                    int p2 = 0;
                    pack_node(btopw, nsz, p2, c);
                    MPI_Isend(btopw, p2, MPI_PACKED, 0, TAG_TOPW2M,
                              MPI_COMM_WORLD, &rqtopw);
                }
            } else if (!PilaLlena(&stack)) {
                PilaPush(&stack, c);
            }
        }
    }

    // ----------------- Cierre ordenado del worker -----------------
    // 1) Vaciar (flush) los envios asincronos propios pendientes
    if (rqneed != MPI_REQUEST_NULL) MPI_Wait(&rqneed, MPI_STATUS_IGNORE);
    if (rqtopw != MPI_REQUEST_NULL) MPI_Wait(&rqtopw, MPI_STATUS_IGNORE);
    // 2) Confirmar al master que ya no enviara nada mas
    int ack = 1;
    MPI_Send(&ack, 1, MPI_INT, 0, TAG_TERMACK, MPI_COMM_WORLD);
    // 3) Cancelar los Irecv persistentes y liberar memoria
    if (rqwork != MPI_REQUEST_NULL) {
        MPI_Cancel(&rqwork);
        MPI_Request_free(&rqwork);
    }
    if (rqtopm != MPI_REQUEST_NULL) {
        MPI_Cancel(&rqtopm);
        MPI_Request_free(&rqtopm);
    }
    if (rqstl != MPI_REQUEST_NULL) {
        MPI_Cancel(&rqstl);
        MPI_Request_free(&rqstl);
    }
    if (rqterm != MPI_REQUEST_NULL) {
        MPI_Cancel(&rqterm);
        MPI_Request_free(&rqterm);
    }

    delete[] bwork;
    delete[] btopm;
    delete[] btopw;
    delete[] bstl;
}

// ===========================================================================
//  MAIN
// ===========================================================================
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, N;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &N);

    if (N < 2) {
        if (rank == 0)
            fprintf(stderr, "Se necesitan >= 2 procesos (1 master + 1 worker)\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        if (argc != 2) {
            fprintf(stderr, "Uso: %s <fichero_matriz>\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        FILE *f = fopen(argv[1], "r");
        if (!f) {
            fprintf(stderr, "Error abriendo %s\n", argv[1]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fscanf(f, "%u", &NCIUDADES);
        fclose(f);
    }
    MPI_Bcast(&NCIUDADES, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    int **tsp = reservarMatrizCuadrada(NCIUDADES);
    if (rank == 0) {
        LeerMatriz(argv[1], tsp);
        if (Inconsistente(tsp)) {
            fprintf(stderr, "Error: matriz TSP inconsistente\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    MPI_Bcast(tsp[0], NCIUDADES * NCIUDADES, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) run_master(N, tsp);
    else run_worker(rank, tsp);

    liberarMatriz(tsp);
    MPI_Finalize();
    return 0;
}
