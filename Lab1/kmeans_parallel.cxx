#include <cstdlib>
#include <cstring>
#include <string>
#include <queue>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <random>

#include <mpi.h>
#include <stdio.h>

using std::cin;
using std::cerr;
using std::cout;
using std::endl;
using std::string;
using std::ostream;
using std::setw;
using std::right;
using std::left;
using std::fixed;
using std::vector;
using std::priority_queue;
using std::setprecision;
using std::ifstream;
using std::ostringstream;

// For Master process only
int n_clusters;
int n_files;

double *stars = nullptr;

int *assignments = nullptr;

// For all processes
int n_stars;
int comm_sz;  // Number of processes
int my_rank;  // Process rank

double *k_means = nullptr;
double *my_stars = nullptr;

int *cluster_sizes = nullptr;
int *my_assignments = nullptr;
int *assignments_offsets = nullptr;
int *assignments_sizes = nullptr;
int *stars_offsets = nullptr;
int *stars_sizes = nullptr;


double calculate_distance(double x1, double y1, double z1,
                          double x2, double y2, double z2)
{
    double x_dist = x1 - x2;
    double y_dist = y1 - y2;
    double z_dist = z1 - z2;
    double sum = x_dist*x_dist + y_dist*y_dist + z_dist*z_dist;

    return sum;
}

void initialize() {
    // Create Random Number Generator for a distribution from 0 .. n_stars
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, n_stars - 1);

    // Genereate n_clusters number of random numbers to be used as the random
    // initial stars as a mean for each cluster
    int idx;
    for (int i = 0; i < n_clusters; i++) {
        idx = dist(rng);

        k_means[i * 3]     = stars[idx * 3];
        k_means[i * 3 + 1] = stars[idx * 3 + 1];
        k_means[i * 3 + 2] = stars[idx * 3 + 2];
    }
}

void assignment() {
    double x, y, z;
    double mean_x, mean_y, mean_z;
    double min_distance, current_distance;
    int min_mean_idx;
    for (int i = 0; i < assignments_sizes[my_rank]; i++) {
        // Get current star observation coordinates
        x = my_stars[i * 3];
        y = my_stars[i * 3 + 1];
        z = my_stars[i * 3 + 2];

        // Initialize the 0th mean as the minimum, then we can skip it in the
        // for-loop
        min_mean_idx = 0;
        min_distance = calculate_distance(x, y, z, k_means[0], k_means[1], k_means[2]);

        for (int j = 1; j < n_clusters; j++) {
            // Get current centroid coordinates
            mean_x = k_means[j * 3];
            mean_y = k_means[j * 3 + 1];
            mean_z = k_means[j * 3 + 2];

            // Calculate the distance from the current star observation
            // to the current centroid
            current_distance = calculate_distance(x, y, z, mean_x, mean_y, mean_z);

            if (current_distance < min_distance) {
                min_mean_idx = j;
                min_distance = current_distance;
            }
        }

        my_assignments[i] = min_mean_idx;
    }
}

void reset_cluster_sizes() {
    for (int i = 0; i < n_clusters; i++) {
        cluster_sizes[i] = 0;
    }
}

void calculate_cluster_sizes() {
    reset_cluster_sizes();

    for (int i = 0; i < n_stars; i++) {
        cluster_sizes[assignments[i]]++;
    }
}

/**
 * This function performs the Update step of Lloyd's algorithm. This will
 * return a bool stating whether the simulation should continue since the
 * algorithm has not converged yet.
 *
 * @returns bool true if simulation should continue; otherwise false.
 */
int update() {
    calculate_cluster_sizes();

    // Create a one-dimensional array of the cluster averages. Multiplying by 3
    // because we need to keep track of xyz for each cluster
    double *cluster_avgs = new double[n_clusters * 3];
    for (int i = 0, length = n_clusters * 3; i < length; i++) {
        cluster_avgs[i] = 0.0;
    }

    // Loop through all the assignments so that we can first calculate the
    // totals for xyz coordinates in each cluster
    int cluster;
    for (int i = 0; i < n_stars; i++) {
        // Get the cluster that the current star is assigned to
        cluster = assignments[i];

        cluster_avgs[cluster * 3]     += stars[i * 3];
        cluster_avgs[cluster * 3 + 1] += stars[i * 3 + 1];
        cluster_avgs[cluster * 3 + 2] += stars[i * 3 + 2];
    }

    // Loop through all the clusters and divide the xyz totals by the cluster
    // size to get the final xyz averages for each cluster
    int cluster_size;
    for (int i = 0; i < n_clusters; i++) {
        // Get the current cluster's size
        cluster_size = cluster_sizes[i];

        cluster_avgs[i * 3]     /= cluster_size;
        cluster_avgs[i * 3 + 1] /= cluster_size;
        cluster_avgs[i * 3 + 2] /= cluster_size;
    }

    // Compare cluster_avgs with current k_means to see if new centroids are
    // different from previous ones by some constant (e.g. .0001)
    const double diff_constant = 0.0001;
    double diff_x, diff_y, diff_z;
    bool should_continue = 0;
    for (int i = 0; i < n_clusters; i++) {
        // Get absolute value of the differences of xyz between
        // old means (kmeans) & new means (cluster_avgs)
        diff_x = fabs(cluster_avgs[i * 3]     - k_means[i * 3]);
        diff_y = fabs(cluster_avgs[i * 3 + 1] - k_means[i * 3 + 1]);
        diff_z = fabs(cluster_avgs[i * 3 + 2] - k_means[i * 3 + 2]);

        // Check if any of xyz differ greater than the defined constant.
        // Only one coordinate for one cluster needs to be different for us
        // to continue.
        if (diff_x > diff_constant || diff_y > diff_constant || diff_z > diff_constant) {
            should_continue = 1;
            break;
        }
    }

    memcpy(k_means, cluster_avgs, sizeof(double) * n_clusters * 3);

    delete[] cluster_avgs;

    return should_continue;
}

void output_centroids(int iter) {
    cout << "For iteration " << iter << "\n-----------------" << endl;
    for (int i = 0; i < n_clusters; i++) {
        cout << "\tCentroid for cluster " << i << ": " << setprecision(8)
             << setw(12) << k_means[i * 3]     << " "
             << setw(12) << k_means[i * 3 + 1] << " "
             << setw(12) << k_means[i * 3 + 2] << endl;
    }
    cout << endl;
}

void run_simulation() {
    // Only Master process
    if (my_rank == 0) {
        // Initialize k_means
        initialize();

        output_centroids(0);
    }

    int iteration = 1;
    int should_continue;
    do {
        // Broadcast centroids out to all processes
        MPI_Bcast(
            k_means,         // the data we're broadcasting
            n_clusters * 3,  // the data size
            MPI_DOUBLE,      // the data type
            0,               // the process we're broadcasting from
            MPI_COMM_WORLD
        );

        // Assignment
        assignment();

        MPI_Gatherv(
            my_assignments,              // the data we're sending
            assignments_sizes[my_rank],  // the size of the data we're sending
            MPI_INT,                     // the data type we're sending
            assignments,                 // where we're receiving the data
            assignments_sizes,           // the amount of data we're receiving from each process
            assignments_offsets,         // the starting point for where we receive the data from each process
            MPI_INT,                     // the data type we're receiving
            0,                           // the process we're sending to
            MPI_COMM_WORLD
        );

        // Only Master process
        if (my_rank == 0) {
            // Update
            should_continue = update();

            output_centroids(iteration);
        }

        // Master process needs to broadcast to all processes whether they
        // should continue or not
        MPI_Bcast(
            &should_continue,
            1,
            MPI_INT,
            0,
            MPI_COMM_WORLD
        );

        iteration++;
    } while (should_continue);
}

void usage(char *executable) {
    cerr << "Usage for kmeans:" << endl;
    cerr << "    " << executable << " <argument list>" << endl;
    cerr << "Possible arguments:" << endl;
    cerr << "    num_clusters <int>  : number of clusters to use in the kmeans algorithm" << endl;
    cerr << "    star_files <str>*   : files containing the stars (in LBR coordinates) followed by a cluster identifier (space separated)" << endl;
    exit(1);
}

int main(int argc, char** argv) {
    int star_count;
    string filename;
    vector<string> star_files;

    // Initialize the command-line arguments on every process
    MPI_Init(&argc, &argv);

    // Get the number of processes in MPI_COMM_WORLD, and put it in the
    // 'comm_sz' variable; ie., how many processes are running this program
    // (the same as what you put in the -np argument).
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    // Get the rank of this particular process in MPI_COMM_WORLD, and put it in
    // the 'my_rank' variable -- ie., what number is this process
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    /** Get command-line arguments and count total number of stars ***********/

    // Get first command-line arg as number of clusters
    n_clusters = atoi(argv[1]);

    // Only the master process
    if (my_rank == 0) {
        // Get the rest of command-line args as star files
        for (int i = 2; i < argc; i++) {
            filename = string(argv[i]);

            // First, read the number of stars for each file and add to
            // total counter
            ifstream star_stream(filename.c_str());
            star_stream >> star_count;
            n_stars += star_count;
            star_stream.close();

            if (star_count <= 0) {
                cerr << "Incorrectly formatted star file: '" << filename << "'" << endl;
                cerr << "First line should contain the number of stars in the file, and be > 0." << endl;
                exit(1);
            }

            // Lastly, add filename to vector
            star_files.push_back(filename);
        }

        n_files = star_files.size();

        if (n_files == 0) {
            cerr << "ERROR: star file not specified." << endl;
            usage(argv[0]);
        }

        cout << "Arguments succesfully parsed." << endl;
        cout << "    number of clusters:     " << setw(10) << n_clusters << endl;
        cout << "    total number of stars:  " << setw(10) << n_stars << endl;
        cout << "    star files:    " << endl;
        for (size_t i = 0; i < star_files.size(); i++) {
            cout << "        '" << star_files.at(i) << "'" << endl;
        }
        cout << endl;
    }

    /** Open each file and store all stars into a single one-dimensional array */

    // Allocate arrays used by all processes
    k_means = new double[n_clusters * 3];
    cluster_sizes = new int[n_clusters];
    assignments_offsets = new int[comm_sz];
    assignments_sizes = new int[comm_sz];
    stars_offsets = new int[comm_sz];
    stars_sizes = new int[comm_sz];

    // Only the Master process
    if (my_rank == 0) {
        // Allocate arrays, we are going to put all the stars coordinates into a
        // one-dimenisonal array
        stars = new double[n_stars * 3];
        assignments = new int[n_stars];

        double l, b, r;
        for (int j = 0, current_star = 0; j < n_files; j++) {
            ifstream star_stream(star_files.at(j).c_str());

            // Get number of stars in current file
            star_stream >> star_count;

            for (int i = 0; i < star_count; i++) {
                star_stream >> l >> b >> r;

                // Convert degrees to radians
                l = l * M_PI / 180;
                b = b * M_PI / 180;

                // Convert l b r (galactic) to x y z (cartesian)
                stars[current_star * 3]     = r * cos(b) * sin(l);
                stars[current_star * 3 + 1] = 4.2 - r * cos(l) * cos(b);
                stars[current_star * 3 + 2] = r * sin(b);

                current_star++;
            }

            star_stream.close();
        }
    }
    
    // Broadcast number of stars out to all processes
    MPI_Bcast(
        &n_stars,  // the data we're broadcasting
        1,         // the data size
        MPI_INT,   // the data type
        0,         // the process we're broadcasting from
        MPI_COMM_WORLD
    );

    // Calculate the array slice sizes and offsets for each process
    double count = 0.0;
    int prev_count;
    double stars_to_comm_sz_ratio = (double)n_stars / (double)comm_sz;

    for (int i = 0; i < comm_sz; i++) {
        assignments_offsets[i] = (int)count;

        prev_count = count;

        count += stars_to_comm_sz_ratio;
        assignments_sizes[i] = (int)count - (int)prev_count;
    }
    
    for (int i = 0; i < comm_sz; i++) {
        stars_offsets[i] = assignments_offsets[i] * 3;
        stars_sizes[i] = assignments_sizes[i] * 3;
    }

    // Allocate arrays for each process
    my_stars = new double[assignments_sizes[my_rank] * 3];
    my_assignments = new int[assignments_sizes[my_rank]];
    
    for (int i = 0; i < assignments_sizes[my_rank]; i++) {
        my_stars[i * 3]     = 0.0;
        my_stars[i * 3 + 1] = 0.0;
        my_stars[i * 3 + 2] = 0.0;
        
        my_assignments[i] = 0;
    }

    // Scatter stars to every process
    MPI_Scatterv(
        stars,                 // the data we're scattering
        stars_sizes,           // the size of the data we're scattering to each process
        stars_offsets,         // where the data is going to be sent from in the array to each process
        MPI_DOUBLE,            // the data type we're sending
        my_stars,              // where we're receiving the data
        stars_sizes[my_rank],  // the amount of data we're receiving per process
        MPI_DOUBLE,            // the data type we're receiving
        0,                     // the process we're sending from
        MPI_COMM_WORLD
    );

    run_simulation();

    if (my_rank == 0) {
        delete[] stars;
        delete[] assignments;
    }

    delete[] k_means;
    delete[] cluster_sizes;
    delete[] assignments_offsets;
    delete[] assignments_sizes;
    delete[] stars_offsets;
    delete[] stars_sizes;

    delete[] my_stars;
    delete[] my_assignments;

    MPI_Finalize();

    return 0;
}
