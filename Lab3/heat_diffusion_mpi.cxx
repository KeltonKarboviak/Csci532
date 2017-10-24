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

//    my_block_height = height / v_slices;
//    my_block_width = width / h_slices;

    printf("[%d]: y = %d, x = %d, height = %d, width = %d\n", my_rank, my_block_y, my_block_x, block_heights[my_block_y], block_widths[my_block_x]);

    double **my_values = new double*[height];
    double **my_values_next = new double*[height];

    double *my_packed_values = new double[width*height];

    // Initialize all values to 0
    for (uint32_t i = 0; i < height; i++) {
        my_values[i] = new double[width];
        my_values_next[i] = new double[width];
        for (uint32_t j = 0; j < width; j++) {
            my_values[i][j] = 0.0;
            my_values_next[i][j] = 0.0;
        }
    }

    // Put a heat source on the left column of the simulation
    for (uint32_t i = 0; i < height; i++) {
        my_values[i][0] = 1.0;
        my_values_next[i][0] = 1.0;
    }

    // Put a cold source on the right column of the simulation
    for (uint32_t i = 0; i < height; i++) {
        my_values[i][width - 1] = -1.0;
        my_values_next[i][width - 1] = -1.0;
    }

    if (my_rank == 0)
        write_simulation_state(simulation_name, height, width, 0, my_values);

    // Update the heat values at each step of the simulation for all
    // internal values
//    for (uint32_t time_step = 1; time_step <= time_steps; time_step++) {
//        // The border values are sources/sinks of heat
//        for (uint32_t i = 1; i < height - 1; i++) {
//            for (uint32_t j = 1; j < width - 1; j++) {
//                double up = values[i - 1][j];
//                double down = values[i + 1][j];
//                double left = values[i][j - 1];
//                double right = values[i][j + 1];
//
//                // Set the values of the next time step of the heat simulation
//                values_next[i][j] = (up + down + left + right) / 4.0;
//            }
//        }
//
//        // Swap the values arrays
//        double **temp = values_next;
//        values_next = values;
//        values = temp;
//
//        // Store the simulation state so you can compare this to your MPI version
//        write_simulation_state(simulation_name, height, width, time_step, values);
//    }

    for (uint32_t i = 0; i < height; i++) {
        delete[] my_values[i];
        delete[] my_values_next[i];
    }

    delete[] my_values;
    delete[] my_values_next;
    delete[] my_packed_values;

    MPI_Finalize();

    return 0;
}
