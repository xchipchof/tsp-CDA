//
// Created by xhip on 1/5/26.
//

#include <stdio.h>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <mpi.h>
#include "libtsp.h"

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
        for (int i = 0; i < NCIUDADES; i++) for (int j = 0; j < NCIUDADES; j++) tsp_copia[i][j] = tsp[i][j];
        Reduce(tsp_copia, &ci_inicial);
        liberarMatriz(tsp_copia);
        root.ci = ci_inicial;

        generate_work(N, tsp, &root, &left, &right, top_ten, CS, &por_procesar);
    }

    InicNodo(&sol);
    sol.id = 0;
    TotalNodos--;

    MPI_Bcast(tsp[0], NCIUDADES, MPI_INT, 0, MPI_COMM_WORLD);

    int size_long, size_int, size_incl, size_dest;
    MPI_Pack_size(1, MPI_LONG, MPI_COMM_WORLD, &size_long); // id
    MPI_Pack_size(2, MPI_INT, MPI_COMM_WORLD, &size_int); // ci y orig_excl
    MPI_Pack_size(NCIUDADES, MPI_INT, MPI_COMM_WORLD, &size_incl);
    MPI_Pack_size(NCIUDADES - 2, MPI_INT, MPI_COMM_WORLD, &size_dest);

    int size_per_nodo = size_long + size_int + size_incl + size_dest;

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
        int position = 0;
        tNodo temp;
        InicNodo(&temp);

        while (PilaPop(&por_procesar, &temp)) {
            MPI_Pack(&temp.id, 1, MPI_LONG, sendbuf, total_nodos_iniciales * size_per_nodo, &position, MPI_COMM_WORLD);
            MPI_Pack(&temp.ci, 1, MPI_INT, sendbuf, total_nodos_iniciales * size_per_nodo, &position, MPI_COMM_WORLD);
            MPI_Pack(&temp.orig_excl, 1, MPI_INT, sendbuf, total_nodos_iniciales * size_per_nodo, &position,
                     MPI_COMM_WORLD);
            MPI_Pack(temp.incl, NCIUDADES, MPI_INT, sendbuf, total_nodos_iniciales * size_per_nodo, &position,
                     MPI_COMM_WORLD);
            MPI_Pack(temp.dest_excl, NCIUDADES - 2, MPI_INT, sendbuf, total_nodos_iniciales * size_per_nodo, &position,
                     MPI_COMM_WORLD);
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
        MPI_Unpack(recvbuf, mis_nodos * size_per_nodo, &position_unpack, &nuevo_nodo.id, 1, MPI_LONG, MPI_COMM_WORLD);

        MPI_Unpack(recvbuf, mis_nodos * size_per_nodo, &position_unpack, &nuevo_nodo.ci, 1, MPI_INT, MPI_COMM_WORLD);

        MPI_Unpack(recvbuf, mis_nodos * size_per_nodo, &position_unpack, &nuevo_nodo.orig_excl, 1, MPI_INT,
                   MPI_COMM_WORLD);

        MPI_Unpack(recvbuf, mis_nodos * size_per_nodo, &position_unpack, nuevo_nodo.incl, NCIUDADES, MPI_INT,
                   MPI_COMM_WORLD);

        MPI_Unpack(recvbuf, mis_nodos * size_per_nodo, &position_unpack, nuevo_nodo.dest_excl, NCIUDADES - 2, MPI_INT,
                   MPI_COMM_WORLD);

        PilaPush(&por_procesar, &nuevo_nodo);
    }

    if (rank == 0) {
        delete[] sendbuf;
        delete[] recvbuf;
        delete[] sendcounts;
        delete[] displs;
    }

    bool working = true;
    // processing:
    while (!PilaVacia(&por_procesar) && working) {
        // procesamiento habitual

        // checkear si hay updates del top 10
            // mod el top 10 si hay updates

        // contrastar con el top 10
            // si es mejor que el CS, entonces update local y aviso global

        // checkear por work requests.
            // si hay, PilaDivide() y se purga la segunda pila para enviarlos en MPI_Isend()

        // Si la pila está vacia, entonces hacer requests a otros procesos para recibir más trabajo.
    }


}
