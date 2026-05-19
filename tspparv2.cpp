//
// Arquitectura work pool optimizada con solapamiento real (Double-Buffering)
// Rank 0 = maestro, Rank 1..N-1 = trabajadores.
//

#include "libtsp.h"
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <stdio.h>

// Tags
#define TAG_SOLUCION   40  // entre workers
#define TAG_PETICION   50  // worker -> master
#define TAG_TRABAJO    60  // master -> worker
#define TAG_RESULTADO  70  // worker -> master
#define TAG_TERMINAR   80  // master -> worker

using namespace std;

tNodo *update_top_ten(const tNodo *newBest, tNodo ranking[10]) {
    int new_pos = 0;
    for (int i = 0; i < 10; i++) {
        if (ranking[i].ci > newBest->ci) {
            new_pos = i;
        } else
            break;
    }
    ranking[new_pos] = *newBest;
    return &ranking[0];
}

void generate_work(const int N, int **tsp, tNodo *root, tNodo *left,
                   tNodo *right, tNodo top_ten[10], const tNodo *CS,
                   tPila *por_procesar) {
    while (PilaTamanio(por_procesar) < 2 * N) {
        Ramifica(root, left, right, tsp);

        if (!PilaPush(por_procesar, right)) {
            printf("Error: pila agotada\n");
            liberarMatriz(tsp);
            exit(1);
        }
        if (!PilaPush(por_procesar, left)) {
            printf("Error: pila agotada\n");
            liberarMatriz(tsp);
            exit(1);
        }
        if (CS != nullptr && right->ci < CS->ci)
            CS = update_top_ten(right, top_ten);
        if (CS != nullptr && left->ci < CS->ci)
            CS = update_top_ten(left, top_ten);

        PilaPop(por_procesar, root);
    }
}

int get_pack_size() {
    int size_int, size_long, size_incl, size_dest;
    MPI_Pack_size(1, MPI_LONG, MPI_COMM_WORLD, &size_long);
    MPI_Pack_size(2, MPI_INT, MPI_COMM_WORLD, &size_int);
    MPI_Pack_size(NCIUDADES, MPI_INT, MPI_COMM_WORLD, &size_incl);
    MPI_Pack_size(NCIUDADES - 2, MPI_INT, MPI_COMM_WORLD, &size_dest);
    return size_long + size_int + size_incl + size_dest;
}

void pack_node(int buffer_size, char *sendbuf, int &position,
               const tNodo *temp) {
    MPI_Pack(&temp->id, 1, MPI_LONG, sendbuf, buffer_size, &position,
             MPI_COMM_WORLD);
    MPI_Pack(&temp->ci, 1, MPI_INT, sendbuf, buffer_size, &position,
             MPI_COMM_WORLD);
    MPI_Pack(&temp->orig_excl, 1, MPI_INT, sendbuf, buffer_size, &position,
             MPI_COMM_WORLD);
    MPI_Pack(temp->incl, NCIUDADES, MPI_INT, sendbuf, buffer_size, &position,
             MPI_COMM_WORLD);
    MPI_Pack(temp->dest_excl, NCIUDADES - 2, MPI_INT, sendbuf, buffer_size,
             &position, MPI_COMM_WORLD);
}

void unpack_node(int buffer_size, char *const recvbuf, int &position_unpack,
                 tNodo *nuevo_nodo) {
    MPI_Unpack(recvbuf, buffer_size, &position_unpack, &nuevo_nodo->id, 1,
               MPI_LONG, MPI_COMM_WORLD);
    MPI_Unpack(recvbuf, buffer_size, &position_unpack, &nuevo_nodo->ci, 1,
               MPI_INT, MPI_COMM_WORLD);
    MPI_Unpack(recvbuf, buffer_size, &position_unpack, &nuevo_nodo->orig_excl, 1,
               MPI_INT, MPI_COMM_WORLD);
    MPI_Unpack(recvbuf, buffer_size, &position_unpack, nuevo_nodo->incl,
               NCIUDADES, MPI_INT, MPI_COMM_WORLD);
    MPI_Unpack(recvbuf, buffer_size, &position_unpack, nuevo_nodo->dest_excl,
               NCIUDADES - 2, MPI_INT, MPI_COMM_WORLD);
}


// Helper para envíos asíncronos
static void isend_node_to_worker(int dest, const tNodo *nodo,
                                 int buf_size, char *buf,
                                 MPI_Request *req) {
    int pos = 0;
    pack_node(buf_size, buf, pos, nodo);
    MPI_Isend(buf, pos, MPI_PACKED, dest, TAG_TRABAJO, MPI_COMM_WORLD, req);
}

static void send_terminar(int dest) {
    int dummy = 0;
    MPI_Send(&dummy, 1, MPI_INT, dest, TAG_TERMINAR, MPI_COMM_WORLD);
}


// bucle del master
static void run_master(const int N, int **tsp) {

    tNodo root, left, right;    // nodos para la expansión
    tNodo top_ten[10];  // top ten mejores soluciones
    const tNodo *CS = &top_ten[0];  // peor mejor solución del top ten
    tPila pool; // bolsa de trabajos

    InicNodo(&root);
    PilaInic(&pool);

    for (int i = 0; i < 10; i++) InicNodo(&top_ten[i]);

    // preparación de la cópia de la matriz tsp original
    int ci_inicial = 0;
    int **tsp_copia = reservarMatrizCuadrada(NCIUDADES);
    for (int i = 0; i < NCIUDADES; i++)
        for (int j = 0; j < NCIUDADES; j++)
            tsp_copia[i][j] = tsp[i][j];
    Reduce(tsp_copia, &ci_inicial);
    liberarMatriz(tsp_copia);
    root.ci = ci_inicial;


    int num_workers = N - 1;    // TODO: Subirlo a N para tener un worker extra en el procesador del master
    generate_work(num_workers, tsp, &root, &left, &right, top_ten, CS, &pool);

    int buf_size = get_pack_size();

    // Buffers y requests de envío por trabajador
    auto canal_envio_buffers_1 = new char *[N];
    auto canal_envio_buffers_2 = new char *[N];

    MPI_Request *reqs_canal_1 = new MPI_Request[N];
    MPI_Request *reqs_canal_2 = new MPI_Request[N];
    int *worker_busy = new int[N];

    // inicialización de los buffers de envío per worker
    for (int i = 1; i < N; i++) {
        canal_envio_buffers_1[i] = new char[buf_size];
        canal_envio_buffers_2[i] = new char[buf_size];

        reqs_canal_1[i] = MPI_REQUEST_NULL;
        reqs_canal_2[i] = MPI_REQUEST_NULL;
        worker_busy[i] = 0;
    }


    const int res_buf_size = 2 * buf_size + sizeof(int); // dos tareas más el num_hijos del worker
    auto recv_bufs = new char *[N];
    MPI_Request *reqs_recv = new MPI_Request[N];

    for (int i = 1; i < N; i++) {
        recv_bufs[i] = new char[res_buf_size];
        reqs_recv[i] = MPI_REQUEST_NULL;
    }

    // se solapan los irecvs de los resultados de cada worker
    for (int i = 1; i < N; i++) {
        MPI_Irecv(recv_bufs[i], res_buf_size, MPI_PACKED, i, TAG_RESULTADO,
                  MPI_COMM_WORLD, &reqs_recv[i]);
    }

    // se recibe cualquier nueva solución que pueda llegar (asíncronamente)
    char *sol_buf = new char[buf_size];
    MPI_Request req_sol = MPI_REQUEST_NULL;
    MPI_Irecv(sol_buf, buf_size, MPI_PACKED, MPI_ANY_SOURCE, TAG_SOLUCION,
              MPI_COMM_WORLD, &req_sol);

    int trabajadores_activos = 0;

    // Bucle principal del master
    while (true) {
        for (int w = 1; w < N; w++) { // por cada worker
            if (worker_busy[w] >= 2) continue; // tiene trabajo -> pasa de largo
            if (PilaVacia(&pool)) break; // el master no tiene más trabajo -> termina

            tNodo nodo_a_enviar;
            PilaPop(&pool, &nodo_a_enviar);

            if (worker_busy[w] == 0) {
                if (reqs_canal_1[w] != MPI_REQUEST_NULL)
                    MPI_Wait(&reqs_canal_1[w], MPI_STATUS_IGNORE);
                isend_node_to_worker(w, &nodo_a_enviar, buf_size, canal_envio_buffers_1[w], &reqs_canal_1[w]);
            } else {
                if (reqs_canal_2[w] != MPI_REQUEST_NULL)
                    MPI_Wait(&reqs_canal_2[w], MPI_STATUS_IGNORE);
                isend_node_to_worker(w, &nodo_a_enviar, buf_size, canal_envio_buffers_2[w], &reqs_canal_2[w]);
            }


            // Se envia un nodo al worker que no tenga trabajo
            // (se solapa porque el worker tenía dos tareas, una que está procesando
            // y otra que está esperando)
            isend_node_to_worker(w, &nodo_a_enviar, buf_size, canal_envio_buffers_1[w], &reqs_canal_1[w]);
            worker_busy[w]++;
            trabajadores_activos++;
        }

        bool todo_vacio = true;
        for (int i =1; i < N; i++) {
            if (worker_busy[i] > 0) {
                todo_vacio = false;
                break;
            }
        }
        // Si la bolsa está vacía y no hay trabajadores activos el bucle acaba
        if (PilaVacia(&pool) && todo_vacio) break;

        // Comprobar si ha llegado algún update con una mejor solución
        {
            int flag = 0;
            MPI_Test(&req_sol, &flag, MPI_STATUS_IGNORE);
            if (flag) {
                tNodo nueva;
                InicNodo(&nueva);
                int pos = 0;
                unpack_node(buf_size, sol_buf, pos, &nueva);
                if (CS == nullptr || nueva.ci < CS->ci)
                    CS = update_top_ten(&nueva, top_ten);
                
                MPI_Irecv(sol_buf, buf_size, MPI_PACKED, MPI_ANY_SOURCE, TAG_SOLUCION,
                          MPI_COMM_WORLD, &req_sol);
            }
        }


        int index_ready = -1;
        MPI_Status status_ready;
        // Espera hasta que reciba cualquier cosa
        MPI_Waitany(N, reqs_recv, &index_ready, &status_ready);



        if (index_ready != MPI_UNDEFINED && index_ready > 0) {
            int source_rank = index_ready;
            
            int pos = 0;
            int num_hijos = 0;
            // desempaquetamos el nodo enpaquetado
            MPI_Unpack(recv_bufs[source_rank], res_buf_size, &pos, &num_hijos, 1, MPI_INT, MPI_COMM_WORLD);

            for (int hijo = 0; hijo < num_hijos; hijo++) {
                tNodo nodo_hijo;
                InicNodo(&nodo_hijo);
                unpack_node(res_buf_size, recv_bufs[source_rank], pos, &nodo_hijo);
                
                if (Solucion(&nodo_hijo)) {
                    if (CS == nullptr || nodo_hijo.ci < CS->ci) {
                        CS = update_top_ten(&nodo_hijo, top_ten);
                        
                        // updates asíncronas para que el master no se bloquee
                        for (int j = 1; j < N; j++) {
                            int p2 = 0;
                            char *tmp = new char[buf_size];
                            pack_node(buf_size, tmp, p2, &nodo_hijo);
                            //MPI_Request tmp_req;
                            MPI_Send(tmp, p2, MPI_PACKED, j, TAG_SOLUCION, MPI_COMM_WORLD);
                            //MPI_Request_free(&tmp_req); // Desacoplamos el ciclo de vida del buffer
                            delete[] tmp;
                        }
                    }
                } else {
                    if (CS == nullptr || nodo_hijo.ci < CS->ci)
                        PilaPush(&pool, &nodo_hijo);
                }
            }

            // Se solapa el Irecv para mantener el flujo asíncrono continuo
            MPI_Irecv(recv_bufs[source_rank], res_buf_size, MPI_PACKED, source_rank, TAG_RESULTADO,
                      MPI_COMM_WORLD, &reqs_recv[source_rank]);

            worker_busy[source_rank]--;
            trabajadores_activos--;
        }
    }

    // Se espera a todas las requests que queden pendientes,
    // y se les envia a los workers el mensaje de terminación
    for (int w = 1; w < N; w++) {
        if (reqs_canal_1[w] != MPI_REQUEST_NULL)
            MPI_Wait(&reqs_canal_1[w], MPI_STATUS_IGNORE);
        if (reqs_canal_2[w] != MPI_REQUEST_NULL)
            MPI_Wait(&reqs_canal_2[w], MPI_STATUS_IGNORE);
        send_terminar(w);
    }

    // liberación de lso buffers y requests
    MPI_Cancel(&req_sol);
    MPI_Request_free(&req_sol);
    for (int w = 1; w < N; w++) {
        MPI_Cancel(&reqs_recv[w]);
        MPI_Request_free(&reqs_recv[w]);
    }

    for (int w = 1; w < N; w++) {
        delete[] canal_envio_buffers_1[w];
        delete[] canal_envio_buffers_2[w];
        delete[] recv_bufs[w];
    }
    delete[] canal_envio_buffers_1;
    delete[] canal_envio_buffers_2;
    delete[] recv_bufs;
    delete[] reqs_canal_1;
    delete[] reqs_canal_2;
    delete[] reqs_recv;
    delete[] worker_busy;
    delete[] sol_buf;
}

// bucle del worker
static void run_worker(int rank, int N, int **tsp) {
    int buf_size = get_pack_size();
    int res_buf_size = 2 * buf_size + sizeof(int);

    // Se solapan las comunicaciones recibiendo la siguiente tarea mientras que procesamos la actual
    char *work_bufs[2];
    work_bufs[0] = new char[buf_size]; //  } Ambos del mismo tamaño fijo
    work_bufs[1] = new char[buf_size]; //  }

    MPI_Request req_work[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL}; // inicializadas a null
    int active_buf = 0; // "puntero" que apunta al buffer en el que se procesa en la iteración actual

    char *sol_buf = new char[buf_size];
    MPI_Request req_sol = MPI_REQUEST_NULL;

    // Se lanza las peticiones de trabajo de manera asíncrona
    MPI_Irecv(work_bufs[0], buf_size, MPI_PACKED, 0, TAG_TRABAJO, MPI_COMM_WORLD, &req_work[0]);
    MPI_Irecv(work_bufs[1], buf_size, MPI_PACKED, 0, TAG_TRABAJO, MPI_COMM_WORLD, &req_work[1]);

    MPI_Irecv(sol_buf, buf_size, MPI_PACKED, MPI_ANY_SOURCE, TAG_SOLUCION, MPI_COMM_WORLD, &req_sol);

    int term_dummy = 0;
    MPI_Request req_term = MPI_REQUEST_NULL;
    MPI_Irecv(&term_dummy, 1, MPI_INT, 0, TAG_TERMINAR, MPI_COMM_WORLD, &req_term);

    // Array donde se trackea las 10 mejores soluciones
    tNodo top_ten[10];
    for (int i = 0; i < 10; i++) InicNodo(&top_ten[i]);
    const tNodo *CS = &top_ten[0];  // apuntador al peor nodo de los 10

    char *res_buf = new char[res_buf_size];
    MPI_Request req_res = MPI_REQUEST_NULL;

    bool running = true;

    while (running) {
        // Se comprueba si ha llegado un mensaje de finalización
        {
            int flag = 0;
            MPI_Test(&req_term, &flag, MPI_STATUS_IGNORE);
            if (flag) {
                running = false;
                break;
            }
        }

        // Se comprueba si hay nuevas updates para el top ten
        {
            int flag = 0;
            MPI_Test(&req_sol, &flag, MPI_STATUS_IGNORE);
            if (flag) {
                tNodo nueva;
                InicNodo(&nueva);
                int pos = 0;
                unpack_node(buf_size, sol_buf, pos, &nueva);
                if (CS == nullptr || nueva.ci < CS->ci)
                    CS = update_top_ten(&nueva, top_ten);
                MPI_Irecv(sol_buf, buf_size, MPI_PACKED, MPI_ANY_SOURCE, TAG_SOLUCION,
                          MPI_COMM_WORLD, &req_sol);
            }
        }


        int completed_buf_idx = -1;
        MPI_Status status_work;

        if (req_work[0] == MPI_REQUEST_NULL && req_work[1] == MPI_REQUEST_NULL) {
            running = false;
            break;
        }

        // Alternativa: MPI_Test en un bucle de espera activa que bloquee el procesador
        MPI_Waitany(2, req_work, &completed_buf_idx, &status_work);

        if (completed_buf_idx == MPI_UNDEFINED || completed_buf_idx < 0) {
            running = false;
            break;
        }

        if (completed_buf_idx >= 0) {
            active_buf = completed_buf_idx;

            tNodo nodo;
            InicNodo(&nodo);
            int pos = 0;
            unpack_node(buf_size, work_bufs[active_buf], pos, &nodo);


            // Se solapa el irecv sobre el buffer que acabamos de desocupar
            MPI_Irecv(work_bufs[active_buf], buf_size, MPI_PACKED, 0, TAG_TRABAJO,
                      MPI_COMM_WORLD, &req_work[active_buf]);


            tNodo left, right;
            InicNodo(&left);
            InicNodo(&right);
            Ramifica(&nodo, &left, &right, tsp);

            // Comprobar que el envío de resultados previo ha terminado para evitar
            // sobreescribir y perder los resultados anteriores
            if (req_res != MPI_REQUEST_NULL)
                MPI_Wait(&req_res, MPI_STATUS_IGNORE);

            int res_pos = 0;
            int num_hijos = 0;
            int offset_count = res_pos;
            
            MPI_Pack(&num_hijos, 1, MPI_INT, res_buf, res_buf_size, &res_pos, MPI_COMM_WORLD);

            // lambda para el empaquetado de los nodos hijo
            // También para trackear el número de hijos, así el master sabrá como puede
            // desempaquetarlos correctamente
            auto pack_hijo = [&](tNodo *h) {
                if (CS == nullptr || h->ci < CS->ci) {
                    pack_node(res_buf_size, res_buf, res_pos, h);
                    num_hijos++;
                }
            };
            pack_hijo(&left);
            pack_hijo(&right);

            int tmp_pos = offset_count;
            MPI_Pack(&num_hijos, 1, MPI_INT, res_buf, res_buf_size, &tmp_pos, MPI_COMM_WORLD);

            // Envío asíncrono al master
            MPI_Isend(res_buf, res_pos, MPI_PACKED, 0, TAG_RESULTADO, MPI_COMM_WORLD, &req_res);
        }
    }

    // Liberación de la memória ocupada y de los buffers con requests pendientes.
    if (req_res != MPI_REQUEST_NULL)
        MPI_Wait(&req_res, MPI_STATUS_IGNORE);

    for (int i = 0; i < 2; i++) {
        if (req_work[i] != MPI_REQUEST_NULL) {
            MPI_Cancel(&req_work[i]);
            MPI_Request_free(&req_work[i]);
        }
    }

    if (req_sol != MPI_REQUEST_NULL) {
        MPI_Cancel(&req_sol);
        MPI_Request_free(&req_sol);
    }

    if (req_term != MPI_REQUEST_NULL) {
        MPI_Cancel(&req_term);
        MPI_Request_free(&req_term);
    }

    delete[] work_bufs[0];
    delete[] work_bufs[1];
    delete[] sol_buf;
    delete[] res_buf;
}


int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    const int rank = MPI::COMM_WORLD.Get_rank();
    const int N =  MPI::COMM_WORLD.Get_size();

    if (N < 2) {
        fprintf(stderr, "Se necesitan al menos 2 procesos (1 maestro + 1 trabajador)\n");
        MPI::COMM_WORLD.Abort(1);
    }

    if (rank == 0) {
        if (argc != 2) {
            fprintf(stderr, "Uso: tsppar <archivo>\n");
            MPI::COMM_WORLD.Abort(1);
        }
        FILE *f = fopen(argv[1], "r");
        if (f == nullptr) {
            fprintf(stderr, "Error al abrir el archivo\n");
            MPI::COMM_WORLD.Abort(1);
        }
        fscanf(f, "%d", &NCIUDADES);
        fclose(f);
    }

    // Comm colectiva para pasar el número de ciudades a todos (bloqueante)
    MPI::COMM_WORLD.Bcast(&NCIUDADES, 1, MPI_INT, 0);

    int **tsp = reservarMatrizCuadrada(NCIUDADES);

    if (rank == 0) { // el master lee y procesa la matriz de entrada
        LeerMatriz(argv[1], tsp);
        if (Inconsistente(tsp)) {
            fprintf(stderr, "Error: Matriz TSP inconsistente\n");
            MPI::COMM_WORLD.Abort(1);
        }
    }

    // envío de la matriz ya construida a los workers
    MPI::COMM_WORLD.Bcast(tsp[0], NCIUDADES*NCIUDADES, MPI_INT, 0);

    if (rank == 0)
        run_master(N, tsp);
    else
        run_worker(rank, N, tsp);

    liberarMatriz(tsp);

    MPI::Finalize();
    return 0;
}