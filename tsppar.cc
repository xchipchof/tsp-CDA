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

int main(int argc, char **argv) {
    switch (argc) {
        case 2: {
            const auto f = fopen(argv[1], "r");
            char buf;
            if (fgets(&buf, 1, f) == nullptr) {
                cerr << "Error al leer del archivo" << endl;
                exit(1);
            }
            NCIUDADES = atoi(&buf);
            fclose(f);
            break;
        }

        default: cerr << "La sintaxis es: bbseq <archivo>" << endl;
            exit(1);
            break;
    }
    int **tsp = reservarMatrizCuadrada(NCIUDADES);
    tNodo root, left, right, sol;
    int top_ten[10] = {INFINITO};
    int *CS = &top_ten[0];

    tPila por_procesar;

    MPI::Init(argc, argv);
    const int rank = MPI::COMM_WORLD.Get_rank();
    int N = MPI::COMM_WORLD.Get_size();

    InicNodo(&root);
    PilaInic(&por_procesar);
    InicNodo(&sol);
    sol.id = 0;
    TotalNodos--;

    /*
     * TODO: INIT de los tipos derivados de MPI
     */

    MPI_Datatype

    if (rank == 0) {
        LeerMatriz(argv[1], tsp);
    }
    MPI_Bcast(tsp, NCIUDADES * NCIUDADES, MPI_INT, 0, MPI_COMM_WORLD);
    // Todos los procesos se sincronzan, esperando a que 0 envie la matriz.

    /*
     * TODO: Bcast entre todos, rank 0 envia la matriz inicializada y el resto recibe
     */


    /*
     * TODO: ETC
     */
}
