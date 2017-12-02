#pragma unroll

#define TILE_WIDTH TILE_DEF

int POSITION(int y, int x, int width) {
    return width * y + x;
}

__kernel void matrix_mul(__constant float *A, __constant float *B, __global float *C, __private int shared_side_length) {
    // A's width is the `shared_side_length` variable passed into the function; it's also B's height
    // B's width is the same as C's width (i.e., x_size)

    int x_size = get_global_size(1);

    int global_y = get_global_id(0);
    int global_x = get_global_id(1);

    float result = 0.0;
    for (int i = 0; i < shared_side_length; i++) {
        result += A[shared_side_length * global_y + i] * B[x_size * i + global_x];
    }

    int pos = x_size * global_y + global_x;

    C[pos] = result;
}

__kernel void matrix_mul_tiled(__constant float *A, __constant float *B, __global float *C, __private int shared_side_length) {
    // A's width is the `shared_side_length` variable passed into the function; it's also B's height
    // B's width is the same as C's width (i.e., x_size)

    int x_size = get_global_size(1);

    int global_y = get_global_id(0);
    int global_x = get_global_id(1);

    int local_y = get_local_id(0);
    int local_x = get_local_id(1);

    __local float Ashare[TILE_WIDTH][TILE_WIDTH];
    __local float Bshare[TILE_WIDTH][TILE_WIDTH];

    int A_x, B_y;

    float result = 0.0;
    int outer_loop_length = ceil((float)shared_side_length / TILE_WIDTH);
    for (int m = 0; m < outer_loop_length; m++) {
        // Collectively load into shared memory
        A_x = m * TILE_WIDTH + local_x;
        B_y = m * TILE_WIDTH + local_y;

        // printf("m %d: global[%d][%d]: A_x = %d, B_y = %d\n", m, global_y, global_x, A_x, B_y);

        if (A_x < shared_side_length) {
            Ashare[local_y][local_x] = A[POSITION(global_y, A_x, shared_side_length)];
            // printf("m %d: global[%d][%d]: Ashare[%d][%d] = A[%d][%d] = %f\n",
            //     m, global_y, global_x, local_y, local_x, global_y, A_x, A[POSITION(global_y, A_x, shared_side_length)]);
        }
        if (B_y < shared_side_length) {
            Bshare[local_y][local_x] = B[POSITION(B_y, global_x, x_size)];
            // printf("m %d: global[%d][%d]: Bshare[%d][%d] = B[%d][%d] = %f\n",
            //     m, global_y, global_x, local_y, local_x, B_y, global_x, B[POSITION(B_y, global_x, x_size)]);
        }

        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

        int inner_loop_length = min(TILE_WIDTH, shared_side_length - m * TILE_WIDTH);
        for (int i = 0; i < inner_loop_length; i++) {
            // printf("m %d: global[%d][%d]: Ashare[%d][%d]: %f + Bshare[%d][%d]: %f = %f; result = %f\n",
            //     m, global_y, global_x, local_y, i, Ashare[local_y][i], i, local_x, Bshare[i][local_x], Ashare[local_y][i] + Bshare[i][local_x], result);
            result += Ashare[local_y][i] * Bshare[i][local_x];
        }

        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }

    int pos = x_size * global_y + global_x;

    C[pos] = result;
}
