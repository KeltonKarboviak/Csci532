#include <iomanip>
using std::fixed;
using std::setprecision;
using std::setw;

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

#include <fstream>
using std::ofstream;

#include <sstream>
using std::ostringstream;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include <utility>
using std::pair;
using std::make_pair;

#include <stdio.h>
#include <mpi.h>


// Write the state of the simulation to a file
void write_simulation_state(string name, uint32_t height, uint32_t width, uint32_t time_step, double **values) {
    ostringstream filename;
    filename << name << "_" << height << "x" << width << "_" << time_step;

    ofstream outfile(filename.str());

    for (uint32_t i = 0; i < height; i++) {
        for (uint32_t j = 0; j < width; j++) {
            outfile << setw(10) << fixed << setprecision(5) << values[i][j];
        }
        outfile << endl;
    }
}


void unpack_master_values(double *packed, uint32_t n_processes, uint32_t v_slices, pair<int, int> *hw_pairs, double **unpacked) {
    pair<int, int> hw_pair;
    uint32_t h, w; //, block_y, block_x;
    uint32_t prior_column_height = 0, prior_row_width = 0;

    uint32_t start_x, start_y;
    uint32_t p = 0;
    for (uint32_t i = 0; i < n_processes; i++) {
        hw_pair = hw_pairs[i];

        h = hw_pair.first;
        w = hw_pair.second;

        start_y = prior_column_height;

//        printf("i %d, offset %d, h %d, w %d\n", i, offset, h, w);

        for (uint32_t y = 0; y < h; y++, start_y++) {
            start_x = prior_row_width;
            for (uint32_t x = 0; x < w; x++, start_x++) {
                unpacked[start_y][start_x] = packed[p++];
//                printf("\ty %d, x %d, p %d, prior_h %d, prior_w %d = %lf -> %lf\n",
//                    y, x, p, prior_column_height, prior_row_width, packed[p++], unpacked[y + prior_column_height][x + prior_row_width]);
            }
        }

        // Add current block's width to the prior width
        prior_row_width += w;

        if ((i + 1) % v_slices == 0) {
            // We need to reset the prior block width at the start of every
            // new row, and we also need to add the previous row's height to
            // the prior column height at the start of a new row
            prior_row_width = 0;
            prior_column_height += h;
        }
    }
}


void pack(double **unpacked, uint32_t start_y, uint32_t start_x, uint32_t height, uint32_t width, double *packed) {
    uint32_t i = 0;
    uint32_t y_iters = start_y + height;  // Number of rows to iterate over, excludes halo(s) on top and/or bottom
    uint32_t x_iters = start_x + width;   // Number of columns to iterate over, excludes halo(s) on left and/or right
    for (uint32_t y = start_y; y < y_iters; y++) {
        for (uint32_t x = start_x; x < x_iters; x++) {
            packed[i++] = unpacked[y][x];
        }
    }
}


void col_to_row(double **src, double *dest, int col_num, int length) {
    for (int y = 0; y < length; y++) {
        dest[y] = src[y][col_num];
    }
}


void row_to_col(double *src, double **dest, int col_num, int length) {
    for (int i = 0; i < length; i++) {
        dest[i][col_num] = src[i];
    }
}


void usage(char *executable) {
    cerr << "ERROR, incorrect arguments." << endl
         << "usage:" << endl
         << "\t" << executable << " <simulation name : string> <height : int> <width : int> <vertical slices : int> <horizontal slices : int> <time steps : int>" << endl;
    exit(1);
}


int main(int argc, char **argv) {
    uint32_t height, width;
    uint32_t v_slices, h_slices;
    uint32_t time_steps;

    uint32_t my_block_x, my_block_y;
    uint32_t my_block_height, my_block_width;
    uint32_t my_halo_height, my_halo_width;
    uint32_t pack_start_y, pack_start_x;

    double *my_send_left, *my_recv_left, *my_send_right, *my_recv_right;

    // Process numbers for MPI halo transfers
    int32_t my_p_up = -1, my_p_right = -1, my_p_down = -1, my_p_left = -1;

    // Individual process's block height & width as a std::pair
    pair<int, int> my_block_hw;

    int comm_sz, my_rank;

    // Initialize the command-line arguments on every process
    MPI_Init(&argc, &argv);

    // Get the number of processes in MPI_COMM_WORLD, and put it in the
    // 'comm_sz' variable; ie., how many processes are running this program
    // (the same as what you put in the -np argument).
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    // Get the rank of this particular process in MPI_COMM_WORLD, and put it in
    // the 'my_rank' variable -- ie., what number is this process
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (argc != 7) {
        usage(argv[0]);
    }

    string simulation_name(argv[1]);

    // The height and width of the simulation
    height = atoi(argv[2]);
    width = atoi(argv[3]);

    // Horizontal and vertical slices will be used to determine how to divide up
    // your MPI processes. You will need to have a number of MPI processes equal
    // to v_slices * h_slices
    v_slices = atoi(argv[4]);
    h_slices = atoi(argv[5]);

    if ((uint32_t)comm_sz != v_slices * h_slices) {
        cerr << "This needs to be called with the number of MPI processes equal to vertical slices * horizontal slices" << endl;
        usage(argv[0]);
    }

    // How long to run the simulation for
    time_steps = atoi(argv[6]);

    // Have each process calculate which block is theirs within the larger 2D array
    my_block_y = my_rank / v_slices;
    my_block_x = my_rank % v_slices;

    // Calculate all the block heights & widths
    vector<uint32_t> block_heights(h_slices, height/h_slices);
    vector<uint32_t> block_widths(v_slices, width/v_slices);

    // Add an extra slot to height of the first (height % h_slices) processes
    int remaining_height = height % h_slices;
    for (int i = 0; i < remaining_height; i++) {
        block_heights[i]++;
    }

    // Add an extra slot to width of the first (width % v_slices) processes
    int remaining_width = width % v_slices;
    for (int i = 0; i < remaining_width; i++) {
        block_widths[i]++;
    }

    // Create and initialize an array of height & width pairs for each process
    pair<int, int> *block_hw_pairs = new pair<int, int>[comm_sz];
    int i = 0;
    for (uint32_t y = 0; y < v_slices; y++) {
        for (uint32_t x = 0; x < h_slices; x++) {
            block_hw_pairs[i++] = make_pair(block_heights[y], block_widths[x]);
        }
    }

    my_block_height = block_heights[my_block_y];
    my_block_width = block_widths[my_block_x];
    my_block_hw = block_hw_pairs[my_rank];

    // Calculate an extended height/width for each process that includes
    // their halo
    my_halo_height = my_block_height;
    my_halo_width = my_block_width;
    pack_start_y = 0;
    pack_start_x = 0;

    if (my_block_y != 0) {             // Needs top halo?
        my_halo_height++;
        pack_start_y++;  // Push starting y value down 1
        my_p_up = my_rank - v_slices;
    }

    if (my_block_y != h_slices - 1) {  // Needs bottom halo?
        my_halo_height++;
        my_p_down = my_rank + v_slices;
    }

    if (my_block_x != 0) {             // Needs left halo?
        my_halo_width++;
        pack_start_x++;  // Push starting x value right 1
        my_p_left = my_rank - 1;

        // Allocate arrays for left/right halo transfers
        my_send_left = new double[my_halo_height];
        my_recv_left = new double[my_halo_height];
    }

    if (my_block_x != v_slices - 1) {  // Needs right halo?
        my_halo_width++;
        my_p_right = my_rank + 1;

        // Allocate arrays for left/right halo transfers
        my_send_right = new double[my_halo_height];
        my_recv_right = new double[my_halo_height];
    }

    printf("[%d]: y = %d, x = %d, height = %d, width = %d, pair = (%d, %d), halo = (%d, %d), start = (%d, %d)\n",
        my_rank, my_block_y, my_block_x, my_block_height, my_block_width, my_block_hw.first, my_block_hw.second, my_halo_height, my_halo_width, pack_start_y, pack_start_x);

    /**
     * Initialize the larger 2D array in master process only since it will be
     * handling the printing
     */
    double **master_values;
    double *packed_master_values = new double[height * width];

    if (my_rank == 0) {
        master_values = new double*[height];

        // Initialize all values to 0
        for (uint32_t y = 0; y < height; y++) {
            master_values[y] = new double[width];
            for (uint32_t x = 0; x < width; x++) {
                master_values[y][x] = 0.0;
            }
        }
    }

    /**
     * Initialize individual process's 2D block array
     */

    // Each process's my_values will be created using the halo height & width
    // since we could potentially need to store a column/row from a surrounding
    // process.
    double **my_values = new double*[my_halo_height];
    double **my_values_next = new double*[my_halo_height];

    // However, my_packed_values is created using just the my_block height/width
    // since we will only be packing the process's values excluding the halo
    // column(s)/row(s).
    double *my_packed_values = new double[my_block_height * my_block_width];

    // Initialize all values to 0
    for (uint32_t y = 0; y < my_halo_height; y++) {
        my_values[y] = new double[my_halo_width];
        my_values_next[y] = new double[my_halo_width];
        for (uint32_t x = 0; x < my_halo_width; x++) {
            my_values[y][x] = (double)my_rank;
            my_values_next[y][x] = (double)my_rank;
        }
    }

    // Check if block is on lhs of master 2D array
    if (my_block_x == 0) {
        printf("[%d]: heat source\n", my_rank);
        // Put a heat source on the left column of the simulation
        for (uint32_t y = 0; y < my_halo_height; y++) {
            my_values[y][0] = 1.0;
            my_values_next[y][0] = 1.0;
        }
    }

    // Check if block is on rhs of master 2D array
    if (my_block_x == v_slices - 1) {
        printf("[%d]: cold source\n", my_rank);
        // Put a cold source on the right column of the simulation
        for (uint32_t y = 0; y < my_halo_height; y++) {
            my_values[y][my_halo_width - 1] = -1.0;
            my_values_next[y][my_halo_width - 1] = -1.0;
        }
    }

//    write_simulation_state(simulation_name + std::to_string(my_rank), my_block_height, my_block_width, 0, my_values);

    /**
     * Calculate the slice_sizes and offsets when the master process uses
     * Gatherv to get every process's my_values and unpack them into the
     * master_values array
     */
    int count = 0;
    int *slice_sizes = new int[comm_sz];
    int *offsets = new int[comm_sz];

    for (int i = 0; i < comm_sz; i++) {
        slice_sizes[i] = block_hw_pairs[i].first * block_hw_pairs[i].second;

        offsets[i] = count;

        count += slice_sizes[i];
//        if (my_rank == 0) printf("%d, %d\n", slice_sizes[i], offsets[i]);
    }

    /**
     * Run simulation loop for specified number of time steps
     */
    uint32_t time_step = 0;
    do {
        // First each individual process packs their 2D array block into a 1D array
        // for sending through MPI
        pack(my_values, pack_start_y, pack_start_x, my_block_height, my_block_width, my_packed_values);

        // Send each block to the master process
        MPI_Gatherv(
            my_packed_values,      // the data we're sending
            slice_sizes[my_rank],  // the size of the data we're sending
            MPI_DOUBLE,            // the data type we're sending
            packed_master_values,  // where we're receiving the data
            slice_sizes,           // the amount of data we're receiving from each process
            offsets,               // the starting point of where we receive the data from each process
            MPI_DOUBLE,            // the data type we're receiving
            0,                     // the process we're gathering to
            MPI_COMM_WORLD
        );

        if (my_rank == 0) {
            // Unpack the values received from every other process into
            // master_values
            unpack_master_values(packed_master_values, comm_sz, v_slices, block_hw_pairs, master_values);

            for (uint32_t y = 0; y < height; y++) {
                for (uint32_t x = 0; x < width; x++) {
                    printf("%10.5f", master_values[y][x]);
                }
                printf("\n");
            }
            printf("\n\n");

            write_simulation_state(simulation_name, height, width, time_step, master_values);
        }

        // Transfer halos
        if (my_p_up >= 0) {  // Top halo
            MPI_Send(
                my_values[1],  // Data we're sending
                my_halo_width, // Size of data we're sending
                MPI_DOUBLE,    // Type of data we're sending
                my_p_up,       // Process we're sending data to
                0,             // Tag of message
                MPI_COMM_WORLD
            );

            MPI_Recv(
                my_values[0],  // Where we're receiving data
                my_halo_width, // Size of data we're receiving
                MPI_DOUBLE,    // Type of data we're receiving
                my_p_up,       // Process we're receiving data from
                0,             // Tag of message to receive
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE
            );
        }

        if (my_p_down >= 0) {  // Bottom halo
            MPI_Send(my_values[my_halo_height - 2], my_halo_width, MPI_DOUBLE, my_p_down, 0, MPI_COMM_WORLD);

            MPI_Recv(my_values[my_halo_height - 1], my_halo_width, MPI_DOUBLE, my_p_down, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (my_p_left >= 0) {  // Left halo
            col_to_row(my_values, my_send_left, 1, my_halo_height);
            MPI_Send(my_send_left, my_halo_height, MPI_DOUBLE, my_p_left, 0, MPI_COMM_WORLD);

            MPI_Recv(my_recv_left, my_halo_height, MPI_DOUBLE, my_p_left, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            row_to_col(my_recv_left, my_values, 0, my_halo_height);
        }

        if (my_p_right >= 0) {  // Right halo
            col_to_row(my_values, my_send_right, my_halo_width-2, my_halo_height);
            MPI_Send(my_send_right, my_halo_height, MPI_DOUBLE, my_p_right, 0, MPI_COMM_WORLD);

            MPI_Recv(my_recv_right, my_halo_height, MPI_DOUBLE, my_p_right, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            row_to_col(my_recv_right, my_values, my_halo_width-1, my_halo_height);
        }

        // The border values are either sources/sinks of heat or halo values,
        // so we exclude them from the update
        for (uint32_t y = 1; y < my_halo_height - 1; y++) {
            for (uint32_t x = 1; x < my_halo_width - 1; x++) {
                double up = my_values[y - 1][x];
                double down = my_values[y + 1][x];
                double left = my_values[y][x - 1];
                double right = my_values[y][x + 1];

                // Set the values of the next time step of the heat simulation
                my_values_next[y][x] = (up + down + left + right) / 4.0;
            }
        }

//        if (my_rank == 3) {
//            printf("[%d]----------\n", my_rank);
//            for (uint32_t y = 0; y < my_halo_height; y++) {
//                for (uint32_t x = 0; x < my_halo_width; x++) {
//                    printf("%10.5f", my_values[y][x]);
//                }
//                printf("\n");
//            }
//            printf("\n\n");
//        }

        // Swap the values arrays
        double **temp = my_values_next;
        my_values_next = my_values;
        my_values = temp;
    } while (++time_step <= time_steps);


    /**
     * Cleanup
     */

    // Only the master process
    if (my_rank == 0) {
        for (uint32_t y = 0; y < height; y++) {
            delete[] master_values[y];
        }
        delete[] master_values;
        delete[] packed_master_values;
    }

    for (uint32_t y = 0; y < my_block_height; y++) {
        delete[] my_values[y];
        delete[] my_values_next[y];
    }

    if (my_p_left >= 0) {
        delete[] my_send_left;
        delete[] my_recv_left;
    }

    if (my_p_right >= 0) {
        delete[] my_send_right;
        delete[] my_recv_right;
    }

    delete[] block_hw_pairs;
    delete[] my_values;
    delete[] my_values_next;
    delete[] my_packed_values;

    delete[] slice_sizes;
    delete[] offsets;

    MPI_Finalize();

    return 0;
}
