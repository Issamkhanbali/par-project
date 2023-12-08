#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void rnn_kernel(float *data, float *weights_input, float *weights_hidden, float *bias, int input_size, int hidden_size, int data_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < data_size) {
        int timestep = idx / input_size;
        if (timestep > 0) {
            for (int i = 0; i < hidden_size; ++i) {
                float output = bias[i];
                for (int j = 0; j < input_size; ++j) {
                    output += data[idx + j] * weights_input[j * hidden_size + i];
                }
                for (int j = 0; j < hidden_size; ++j) {
                    output += data[(timestep - 1) * hidden_size + j] * weights_hidden[j * hidden_size + i];
                }
                data[timestep * hidden_size + i] = tanh(output);
            }
        }
    }
}

int main() {
    int input_size = 3;
    int hidden_size = 4;
    int timesteps = 10;
    int data_size = timesteps * input_size;

    float *h_data = (float *)malloc(data_size * sizeof(float));
    float *h_weights_input = (float *)malloc(input_size * hidden_size * sizeof(float));
    float *h_weights_hidden = (float *)malloc(hidden_size * hidden_size * sizeof(float));
    float *h_bias = (float *)malloc(hidden_size * sizeof(float));

   
    for (int i = 0; i < data_size; ++i) {
        h_data[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < input_size * hidden_size; ++i) {
        h_weights_input[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < hidden_size * hidden_size; ++i) {
        h_weights_hidden[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < hidden_size; ++i) {
        h_bias[i] = (float)rand() / RAND_MAX;
    }

    float *d_data, *d_weights_input, *d_weights_hidden, *d_bias;
    cudaMalloc(&d_data, data_size * sizeof(float));
    cudaMalloc(&d_weights_input, input_size * hidden_size * sizeof(float));
    cudaMalloc(&d_weights_hidden, hidden_size * hidden_size * sizeof(float));
    cudaMalloc(&d_bias, hidden_size * sizeof(float));

    cudaMemcpy(d_data, h_data, data_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights_input, h_weights_input, input_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights_hidden, h_weights_hidden, hidden_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, hidden_size * sizeof(float), cudaMemcpyHostToDevice);

    
    int blockSize = 1;  
    int numBlocks = 1;
    rnn_kernel<<<numBlocks, blockSize>>>(d_data, d_weights_input, d_weights_hidden, d_bias, input_size, hidden_size, data_size);

    
    blockSize = 2;
    rnn_kernel<<<numBlocks, blockSize>>>(d_data, d_weights_input, d_weights_hidden, d_bias, input_size, hidden_size, data_size);

   
    blockSize = 4;
    rnn_kernel<<<numBlocks, blockSize>>>(d_data, d_weights_input, d_weights_hidden, d_bias, input_size, hidden_size, data_size);

    cudaMemcpy(h_data, d_data, data_size * sizeof(float), cudaMemcpyDeviceToHost);

    free(h_data);
    free(h_weights_input);
    free(h_weights_hidden);
    free(h_bias);
    cudaFree(d_data);
    cudaFree(d_weights_input);
    cudaFree(d_weights_hidden);
    cudaFree(d_bias);

    return 0;
}
