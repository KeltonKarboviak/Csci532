#include "stdlib.h"
#include "stdio.h"
#include "string.h"

void print_matrix(const char *name, float **matrix, int height, int width) {
    printf("%s\n", name);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            printf(" %5.4f", matrix[y][x]);
        }
        printf("\n");
    }
    printf("\n");
}

void usage(char *executable) {
    printf("ERROR, incorrect arguments.\n");
    printf("usage:\n");
    printf("\t %s <input height: int> <input width: int> <filter height: int> <filter width: int>\n", executable);
    exit(1);
}

int main(int argc, char **argv) {
    if (argc < 5) {
        usage(argv[0]);
    }

    // Hard-code for testing output
    srand48(20171116);

    int input_height = atoi(argv[1]);
    int input_width = atoi(argv[2]);

    int filter_height = atoi(argv[3]);
    int filter_width = atoi(argv[4]);
    
    int no_write = 0;
    if (argc > 5 && !strcmp(argv[5], "--no-write")) {
        no_write = 1;
    }

    int output_height = input_height - filter_height + 1;
    int output_width = input_width - filter_width + 1;

    float **input = (float**)malloc(sizeof(float*) * input_height);
    for (int y = 0; y < input_height; y++) {
        input[y] = (float*)malloc(sizeof(float) * input_width);

        for (int x = 0; x < input_width; x++) {
            input[y][x] = drand48() * 100;
        }
    }

    float **filter = (float**)malloc(sizeof(float*) * filter_height);
    for (int y = 0; y < filter_height; y++) {
        filter[y] = (float*)malloc(sizeof(float) * filter_width);

        for (int x = 0; x < filter_width; x++) {
            filter[y][x] = drand48() * 100;
        }
    }

    float **output = (float**)malloc(sizeof(float*) * output_height);
    for (int y = 0; y < output_height; y++) {
        output[y] = (float*)malloc(sizeof(float) * output_width);

        for (int x = 0; x < output_width; x++) {
            output[y][x] = 0.0;
        }
    }

    if (!no_write) {
        print_matrix("input", input, input_height, input_width);
        print_matrix("filter", filter, filter_height, filter_width);
    }

    for (int y = 0; y < output_height; y++) {
        for (int x = 0; x < output_width; x++) {

            for (int cy = 0; cy < filter_height; cy++) {
                for (int cx = 0; cx < filter_width; cx++) {
                    output[y][x] += input[y + cy][x + cx] * filter[cy][cx];
                }
            }

        }
    }
    
    if (!no_write) {
        print_matrix("output", output, output_height, output_width);
    }
}
