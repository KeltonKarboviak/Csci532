#include "stdlib.h"
#include "stdio.h"

void print_matrix(const char *name, float **matrix, int height, int width) {
    printf("%s\n", name);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            printf(" %5.2f", matrix[y][x]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int n_arguments, char **arguments) {
    int input_height = 16;
    int input_width = 16;

    float **input = (float**)malloc(sizeof(float*) * input_height);
    for (int y = 0; y < input_height; y++) {
        input[y] = (float*)malloc(sizeof(float) * input_width);

        for (int x = 0; x < input_width; x++) {
            input[y][x] = drand48() * 100;
        }
    }

    int filter_height = 5;
    int filter_width = 5;

    float **filter = (float**)malloc(sizeof(float*) * filter_height);
    for (int y = 0; y < filter_height; y++) {
        filter[y] = (float*)malloc(sizeof(float) * filter_width);

        for (int x = 0; x < filter_width; x++) {
            filter[y][x] = drand48() * 100;
        }
    }

    int output_height = input_height - filter_height + 1;
    int output_width = input_width - filter_width + 1;

    float **output = (float**)malloc(sizeof(float*) * output_height);
    for (int y = 0; y < output_height; y++) {
        output[y] = (float*)malloc(sizeof(float) * output_width);

        for (int x = 0; x < output_width; x++) {
            output[y][x] = 0.0;
        }
    }

    print_matrix("input", input, input_height, input_width);
    print_matrix("filter", filter, filter_height, filter_width);

    for (int y = 0; y < output_height; y++) {
        for (int x = 0; x < output_width; x++) {

            for (int cy = 0; cy < filter_height; cy++) {
                for (int cx = 0; cx < filter_width; cx++) {
                    output[y][x] += input[y + cy][x + cx] * filter[cy][cx];
                }
            }

        }
    }

    print_matrix("output", output, output_height, output_width);
}
