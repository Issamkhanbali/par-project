#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    float *weights_input;
    float *weights_hidden;
    float bias;
    int input_size;
    int hidden_size;
} RNNCell;

void init_rnn_cell(RNNCell *cell, int input_size, int hidden_size) {
    cell->input_size = input_size;
    cell->hidden_size = hidden_size;
    cell->weights_input = (float *)malloc(input_size * hidden_size * sizeof(float));
    cell->weights_hidden = (float *)malloc(hidden_size * hidden_size * sizeof(float));
    cell->bias = 0.0;

    for (int i = 0; i < input_size * hidden_size; i++) {
        cell->weights_input[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < hidden_size * hidden_size; i++) {
        cell->weights_hidden[i] = (float)rand() / RAND_MAX;
    }
}

void rnn_cell_forward(RNNCell *cell, float *input, float *hidden_state, float *output) {
    for (int i = 0; i < cell->hidden_size; i++) {
        output[i] = cell->bias;
        for (int j = 0; j < cell->input_size; j++) {
            output[i] += input[j] * cell->weights_input[j * cell->hidden_size + i];
        }
        for (int j = 0; j < cell->hidden_size; j++) {
            output[i] += hidden_state[j] * cell->weights_hidden[j * cell->hidden_size + i];
        }
        output[i] = tanh(output[i]);
    }
}

void rnn_process(float *data, int data_size, int input_size, int hidden_size) {
    RNNCell cell;
    init_rnn_cell(&cell, input_size, hidden_size);

    float *hidden_state = (float *)calloc(hidden_size, sizeof(float));
    float *output = (float *)malloc(hidden_size * sizeof(float));

    for (int i = 0; i < data_size; i += input_size) {
        rnn_cell_forward(&cell, &data[i], hidden_state, output);
        memcpy(hidden_state, output, hidden_size * sizeof(float));
    }

    free(cell.weights_input);
    free(cell.weights_hidden);
    free(hidden_state);
    free(output);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int total_data_size = 10000;
    int input_size = 3;
    int hidden_size = 4;
    int timesteps = total_data_size / input_size;
    int chunk_size = timesteps / world_size;

    float *data = NULL;
    float *sub_data = (float *)malloc(chunk_size * input_size * sizeof(float));

    if (world_rank == 0) {
        data = (float *)malloc(total_data_size * sizeof(float));

    }


    MPI_Scatter(data, chunk_size * input_size, MPI_FLOAT, sub_data, chunk_size * input_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

    rnn_process(sub_data, chunk_size * input_size, input_size, hidden_size);


    free(sub_data);
    if (world_rank == 0) {
        free(data);
    }

    MPI_Finalize();
    return 0;
}
