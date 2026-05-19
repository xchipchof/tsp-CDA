//
// Created by xhip on 1/5/26.
//

#include <stdio.h>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <mpi.h>
#include "libtsp.h"

#define TAG_SOLUCION 40
#define TAG_PETICION 50

using namespace std;

tNodo *update_top_ten(const tNodo *newBest, tNodo ranking[10]) {
    int new_pos = 0;
    for (int i = 0; i < 10; i++) {
        if (ranking[i].ci > newBest->ci) {
            new_pos = i;
        } else break;
    }
    ranking[new_pos] = *newBest;

    return &ranking[0];
}

// Bucle de generación de trabajo inicial
void generate_work(const int N, int **tsp, tNodo *root, tNodo *left, tNodo *right, tNodo top_ten[10], const tNodo *CS,
                   tPila *por_procesar) {
    while (PilaTamanio(por_procesar) < 2 * N) {
        Ramifica(root, left, right, tsp);

        // (Tu lógica de push y update_top_ten se mantiene igual)
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
        if (CS != nullptr && right->ci < CS->ci) {
            //  cota inferior del hijo derecho menor que la peor cota superior del top 10
            CS = update_top_ten(right, top_ten);
        }

        if (CS != nullptr && left->ci < CS->ci) {
            // cota inferior del hijo izquierdo menor que la peor cota superior del top 10
            CS = update_top_ten(left, top_ten);
        }

        PilaPop(por_procesar, root);
    }
}

int get_pack_size() {
    int size_int;
    int size_long;
    int size_incl;
    int size_dest;

    MPI_Pack_size(1, MPI_LONG, MPI_COMM_WORLD, &size_long); // id
    MPI_Pack_size(2, MPI_INT, MPI_COMM_WORLD, &size_int); // ci y orig_excl
    MPI_Pack_size(NCIUDADES, MPI_INT, MPI_COMM_WORLD, &size_incl);
    MPI_Pack_size(NCIUDADES - 2, MPI_INT, MPI_COMM_WORLD, &size_dest);

    return size_long + size_int + size_incl + size_dest;
}

void pack_node(int buffer_size, char *sendbuf, int &position, const tNodo *temp) {
    MPI_Pack(&temp->id, 1, MPI_LONG, sendbuf, buffer_size, &position, MPI_COMM_WORLD);
    MPI_Pack(&temp->ci, 1, MPI_INT, sendbuf, buffer_size, &position, MPI_COMM_WORLD);
    MPI_Pack(&temp->orig_excl, 1, MPI_INT, sendbuf, buffer_size, &position, MPI_COMM_WORLD);
    MPI_Pack(temp->incl, NCIUDADES, MPI_INT, sendbuf, buffer_size, &position, MPI_COMM_WORLD);
    MPI_Pack(temp->dest_excl, NCIUDADES - 2, MPI_INT, sendbuf, buffer_size, &position, MPI_COMM_WORLD);
}

void unpack_node(int buffer_size, char *const recvbuf, int &position_unpack, tNodo *nuevo_nodo) {
    MPI_Unpack(recvbuf, buffer_size, &position_unpack, &nuevo_nodo->id, 1, MPI_LONG, MPI_COMM_WORLD);
    MPI_Unpack(recvbuf, buffer_size, &position_unpack, &nuevo_nodo->ci, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Unpack(recvbuf, buffer_size, &position_unpack, &nuevo_nodo->orig_excl, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Unpack(recvbuf, buffer_size, &position_unpack, nuevo_nodo->incl, NCIUDADES, MPI_INT, MPI_COMM_WORLD);
    MPI_Unpack(recvbuf, buffer_size, &position_unpack, nuevo_nodo->dest_excl, NCIUDADES - 2, MPI_INT, MPI_COMM_WORLD);
}

void send_tmp_stack_to_receptor(tPila *pila, int destino) {
    int num_nodos = PilaTamanio(pila);

    MPI_Send(&num_nodos, 1, MPI_INT, destino, TAG_PETICION, MPI_COMM_WORLD);

    if (num_nodos > 0) {
        int size_per_nodo = get_pack_size();
        int buffer_size = num_nodos * size_per_nodo;
        auto sendbuf = new char[buffer_size];

        int position = 0;
        tNodo temp;
        InicNodo(&temp);

        while (PilaPop(pila, &temp)) {
            pack_node(buffer_size, sendbuf, position, &temp);
        }

        MPI_Send(sendbuf, position, MPI_PACKED, destino, TAG_PETICION + 1, MPI_COMM_WORLD);
        delete[] sendbuf;
    }
}

void send_global_update_for_CS(const tNodo *solucion, int N, int rank) {
    int size_per_nodo = get_pack_size();
    char *sendbuf = new char[size_per_nodo];
    int position = 0;

    pack_node(size_per_nodo, sendbuf, position, solucion);

    for (int i = 0; i < N; i++) {
        if (i != rank) {
            // Utilizamos un tag específico para actualizaciones de solución
            MPI_Send(sendbuf, position, MPI_PACKED, i, TAG_SOLUCION, MPI_COMM_WORLD);
        }
    }
    delete[] sendbuf;
}

void recv_and_update_top_ten(tNodo ranking[10], int rank_emisor) {
    int size_per_nodo = get_pack_size();
    char *recvbuf = new char[size_per_nodo];

    MPI_Recv(recvbuf, size_per_nodo, MPI_PACKED, rank_emisor, TAG_SOLUCION, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    tNodo nueva_solucion;
    InicNodo(&nueva_solucion);
    int position = 0;

    unpack_node(size_per_nodo, recvbuf, position, &nueva_solucion);

    // Actualizamos nuestro ranking local con la solución recibida
    update_top_ten(&nueva_solucion, ranking);
    delete[] recvbuf;
}

void solicitar_trabajo_a_otros(int rank, int N, tPila *pila, bool *working) {
    int target = (rank + 1) % N; // Preguntamos al proceso de al lado (anillo)
    int peticion_vacia = 1;
    MPI_Request peticion_trabajo;

    MPI_Send(&peticion_vacia, 1, MPI_INT, target, TAG_PETICION, MPI_COMM_WORLD);

    int num_nodos_recibidos = 0;
    // Esperamos la respuesta de la cantidad de nodos
    MPI_Recv(&num_nodos_recibidos, 1, MPI_INT, target, TAG_PETICION, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if (num_nodos_recibidos > 0) {
        int size_per_nodo = get_pack_size();
        int buffer_size = num_nodos_recibidos * size_per_nodo;
        char *recvbuf = new char[buffer_size];

        // 3. Recibimos la carga útil empaquetada
        MPI_Recv(recvbuf, buffer_size, MPI_PACKED, target, TAG_PETICION + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        int position = 0;
        for (int i = 0; i < num_nodos_recibidos; i++) {
            tNodo nuevo;
            InicNodo(&nuevo);
            unpack_node(buffer_size, recvbuf, position, &nuevo);
            PilaPush(pila, &nuevo);
        }
        delete[] recvbuf;
    } else {
        *working = false;
    }
}

int main(int argc, char **argv) {
    MPI::Init(argc, argv);
    const int rank = MPI::COMM_WORLD.Get_rank();
    const int N = MPI::COMM_WORLD.Get_size();

    if (rank == 0) {
        if (argc != 2) {
            cerr << "La sintaxis es: bbseq <archivo>" << endl;
            MPI::COMM_WORLD.Abort(1);
        }

        FILE *f = fopen(argv[1], "r");
        if (f == nullptr) {
            cerr << "Error al abrir el archivo" << endl;
            MPI::COMM_WORLD.Abort(1);
        }
        fscanf(f, "%d", &NCIUDADES);
        fclose(f);
    }

    MPI_Bcast(&NCIUDADES, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int **tsp = reservarMatrizCuadrada(NCIUDADES);
    tNodo root, left, right, sol;
    tNodo top_ten[10];
    const tNodo *CS = &top_ten[0];
    tPila por_procesar;

    InicNodo(&root);
    PilaInic(&por_procesar);

    if (rank == 0) {
        LeerMatriz(argv[1], tsp);

        if (Inconsistente(tsp)) {
            printf("Error: Matriz TSP es inconsistente\n");
            MPI::COMM_WORLD.Abort(1);
        }

        int ci_inicial = 0;
        int **tsp_copia = reservarMatrizCuadrada(NCIUDADES);
        for (int i = 0; i < NCIUDADES; i++)
            for (int j = 0; j < NCIUDADES; j++)
                tsp_copia[i][j] = tsp[i][j];
        Reduce(tsp_copia, &ci_inicial);
        liberarMatriz(tsp_copia);
        root.ci = ci_inicial;

        generate_work(N, tsp, &root, &left, &right, top_ten, CS, &por_procesar);
    }

    InicNodo(&sol);
    sol.id = 0;
    TotalNodos--;

    MPI_Bcast(tsp[0], NCIUDADES * NCIUDADES, MPI_INT, 0, MPI_COMM_WORLD);

    int size_per_nodo = get_pack_size();

    int total_nodos_iniciales = 0;
    if (rank == 0) {
        total_nodos_iniciales = PilaTamanio(&por_procesar);
    }
    MPI_Bcast(&total_nodos_iniciales, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int base = total_nodos_iniciales / N;
    int resto = total_nodos_iniciales % N;
    int mis_nodos = base + (rank < resto ? 1 : 0);

    int *sendcounts = new int[N];
    int *displs = new int[N];
    int current_displ = 0;

    for (int i = 0; i < N; i++) {
        int nodos_para_i = base + (i < resto ? 1 : 0);
        sendcounts[i] = nodos_para_i * size_per_nodo;
        displs[i] = current_displ;
        current_displ += sendcounts[i];
    }

    char *sendbuf = nullptr;
    if (rank == 0) {
        sendbuf = new char[total_nodos_iniciales * size_per_nodo];
        tNodo temp;
        InicNodo(&temp);
        int position = 0;

        while (PilaPop(&por_procesar, &temp)) {
            pack_node(size_per_nodo, sendbuf, position, &temp);
        }
    }
    const auto recvbuf = new char[mis_nodos * size_per_nodo];

    MPI_Scatterv(sendbuf, sendcounts, displs, MPI_PACKED,
                 recvbuf, mis_nodos * size_per_nodo, MPI_PACKED,
                 0, MPI_COMM_WORLD);

    int position_unpack = 0;
    for (int i = 0; i < mis_nodos; i++) {
        tNodo nuevo_nodo;
        InicNodo(&nuevo_nodo);
        unpack_node(size_per_nodo, recvbuf, position_unpack, &nuevo_nodo);

        PilaPush(&por_procesar, &nuevo_nodo);
    }

    if (rank == 0) {
        delete[] sendbuf;
    }
    delete[] recvbuf;
    delete[] sendcounts;
    delete[] displs;

    bool working = true;
    MPI_Request request_difusion = MPI_REQUEST_NULL;
    int peticion_trabajo;
    // processing:
    while (!PilaVacia(&por_procesar) && working) {
        // procesamiento habitual
        PilaPop(&por_procesar, &root);
        Ramifica(&root, &right, &left, tsp);

        // checkear si hay updates del top 10
        // mod el top 10 si hay updates
        int hay_nueva_solucion;
        if (hay_nueva_solucion) {
            tNodo nuevo_nodo;
            InicNodo(&nuevo_nodo);
            recv_and_update_top_ten(top_ten, rank);
        }

        // contrastar con el top 10
        // si es mejor que el CS, entonces update local y aviso global
        if (left.ci < CS->ci) {
            if (Solucion(&left)) {
                update_top_ten(&left, top_ten);
                send_global_update_for_CS(&left, N, rank);
            } else PilaPush(&por_procesar, &left);
        }

        if (right.ci < CS->ci) {
            if (Solucion(&right)) {
                update_top_ten(&right, top_ten);
                send_global_update_for_CS(&right, N, rank);
            } else PilaPush(&por_procesar, &right);
        }

        // checkear por work requests.
        // si hay, PilaDivide() y se purga la segunda pila para enviarlos en MPI_Isend()
        int hay_peticion_trabajo;
        MPI_Iprobe(MPI_ANY_SOURCE, TAG_PETICION, MPI_COMM_WORLD, &hay_peticion_trabajo,MPI_STATUS_IGNORE);
        if (hay_peticion_trabajo && PilaTamanio(&por_procesar) >= 2) {
            int rank_receptor;
            MPI_Status status;
            MPI_Recv(&rank_receptor, 1, MPI_INT, MPI_ANY_SOURCE, TAG_PETICION, MPI_COMM_WORLD, &status);

            tPila pila_tmp;
            PilaInic(&pila_tmp);
            PilaDivide(&por_procesar, &pila_tmp);

            send_tmp_stack_to_receptor(&pila_tmp, rank_receptor);
        }

        // Si la pila está vacia, entonces hacer requests a otros procesos para recibir más trabajo.
        if (PilaVacia(&por_procesar)) {
            solicitar_trabajo_a_otros(rank, N, &por_procesar, &working);
        }
    }

    MPI::Finalize();
    exit(0);
}
