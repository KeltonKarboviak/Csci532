#pragma unroll

__kernel void matrix_mul(__constant float *A, __constant float *B, __global float *C) {
    int z_group = get_group_id(0);
    int y_group = get_group_id(1);
    int x_group = get_group_id(2);

    int z_size = get_global_size(0);
    int y_size = get_global_size(1);
    int x_size = get_global_size(2);

    int z = get_global_id(0);
    int y = get_global_id(1);
    int x = get_global_id(2);

    int pos = (z * (y_size * x_size)) + (y * (x_size)) + x;
    C[pos] = A[pos] + B[pos];

    printf("group[%d][%d][%d], sizes[%d][%d][%d], id[%d][%d][%d], pos: %d, %f + %f = %f\n", z_group, y_group, x_group, z_size, y_size, x_size, z, y, x, pos, A[pos], B[pos], C[pos]);
}
