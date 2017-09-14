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

int n_stars = -1;
int n_clusters = -1;
int n_files = -1;

double *stars = nullptr;
double *k_means = nullptr;

int *cluster_sizes = nullptr;
int *assignments = nullptr;


double calculate_distance(double x1, double y1, double z1,
                          double x2, double y2, double z2)
{
    double x_dist = x1 - x2;
    double y_dist = y1 - y2;
    double z_dist = z1 - z2;    
    double sum = x_dist*x_dist + y_dist*y_dist + z_dist*z_dist;

    return sqrt(sum);
}

void initialize() {    
    // Create Random Number Generator for a distribution from 0 .. n_stars
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, n_stars);

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
    for (int i = 0; i < n_stars; i++) {
        // Get current star observation coordinates
        x = stars[i * 3];
        y = stars[i * 3 + 1];
        z = stars[i * 3 + 2];
        
        // Initialize the 0th mean as the minimum, then we can skip it in the
        // for-loop
        min_mean_idx = 0;
        min_distance = calculate_distance(x, y, z, k_means[0], k_means[1], k_means[2]);
    
        for (int j = 1; j < n_clusters; j++) {
            // Get current centroid coordinates
            mean_x = k_means[j * 3];
            mean_y = k_means[j * 3 + 1];
            mean_z = k_means[j * 3 + 2];
            
            // Calculate the squared distance from the currentstar observation
            // to the current centroid
            current_distance = calculate_distance(x, y, z, mean_x, mean_y, mean_z);
            current_distance *= current_distance;
            
            if (current_distance < min_distance) {
                min_mean_idx = j;
                min_distance = current_distance;
            }
        }
        
        assignments[i] = min_mean_idx;
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
bool update() {
    // cout << "\tStarting Calculating Cluster Sizes" << endl;
    calculate_cluster_sizes();
    // cout << "\tFinished Calculating Cluster Sizes" << endl;
    
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
    // different from previous ones by some constant (e.g. .001)
    const double diff_constant = .001;
    bool should_continue = false;
    double diff_x, diff_y, diff_z;
    for (int i = 0; i < n_clusters; i++) {
        // Get absolute value of the differences of xyz between
        // old means (kmeans) & new means (cluster_avgs)
        diff_x = fabs(cluster_avgs[i * 3]     - k_means[i * 3]);
        diff_y = fabs(cluster_avgs[i * 3 + 1] - k_means[i * 3 + 1]);
        diff_z = fabs(cluster_avgs[i * 3 + 2] - k_means[i * 3 + 2]);
        
        cout << "\tDifferences: " << diff_x << " " << diff_y << " " << diff_z << endl;
        
        // Check if any of xyz differ greater than the defined constant.
        // Only one coordinate for one cluster needs to be different for us
        // to continue.
        if (diff_x > diff_constant || diff_y > diff_constant || diff_z > diff_constant) {
            should_continue = true;
            // break;
        }
    }
    
    memcpy(k_means, cluster_avgs, sizeof(double) * n_clusters * 3);
    
     
    /*for (int i = 0; i < n_clusters; i++) {
        k_means[i * 3]     = cluster_avgs[i * 3];
        k_means[i * 3 + 1] = cluster_avgs[i * 3 + 1];
        k_means[i * 3 + 2] = cluster_avgs[i * 3 + 2];
    }*/
    
    delete[] cluster_avgs;
    
    cout << "In Update(), should continue?: " << (should_continue ? "True" : "False") << endl;
    
    return should_continue;
}

void output_centroids(int iter) {
    cout << "For iteration " << iter << "\n-----------------" << endl;
    for (int i = 0; i < n_clusters; i++) {
        cout << "\tCentroid for cluster " << i << ": " << setprecision(8)
             << setw(12) << k_means[i]     << " "
             << setw(12) << k_means[i + 1] << " "
             << setw(12) << k_means[i + 2] << endl;
    }
    cout << endl;
}

void run_simulation() {
    // Initialize k_means
    // cout << "Starting Forgy initialization" << endl;
    initialize();
    // cout << "Finished Forgy initialization" << endl << endl;
    
    output_centroids(0);
    
    int iteration = 1;
    bool should_continue;
    do {
        // Assignment
        // cout << "Starting Assignment" << endl;
        assignment();
        // cout << "Finished Assignment" << endl << endl;
        
        // Update
        // cout << "Starting Update" << endl;
        should_continue = update();
        cout << "In run_simulation(), should continue?: " << (should_continue ? "True" : "False") << endl;
        // cout << "Finished Update" << endl << endl;
        
        output_centroids(iteration);
        
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
    int count;
    string filename;
    vector<string> star_files;
    
    /** Get command-line arguments and count total number of stars ***********/
    
    // Get first command-line arg as number of clusters
    n_clusters = atoi(argv[1]);
    
    // Get the rest of command-line args as star files
    for (int i = 2; i < argc; i++) {
        filename = string(argv[i]);
        
        // First, read the number of stars for each file and add to
        // total counter
        ifstream star_stream(filename.c_str());
        star_stream >> count;
        n_stars += count;
        star_stream.close();
        
        if (count <= 0) {
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
    
    /** End command-line args & star count ***********************************/

    /** Open each file and store all stars into a single one-dimensional array */

    // Allocate array, we are going to put all the stars coordinates into a
    // one-dimenisonal array
    stars = new double[n_stars * 3];
    
    k_means = new double[n_clusters * 3];
    cluster_sizes = new int[n_clusters];
    assignments = new int[n_stars];
    
    int current_star = 0;
    double l, b, r, x, y, z;
    for (int j = 0; j < n_files; j++) {
        ifstream star_stream(star_files.at(j).c_str());

        // Get number of stars in current file
        star_stream >> count;

        cout << "Reading " << count << " stars." << endl;

        for (int i = 0; i < count; i++) {
            star_stream >> l >> b >> r;

            // Convert degrees to radians
            l = l * M_PI / 180;
            b = b * M_PI / 180;

            // Convert l b r (galactic) to x y z (cartesian)
            x = r * cos(b) * sin(l);
            y = 4.2 - r * cos(l) * cos(b);
            z = r * sin(b);
            
            // Assign xyz to our one-dimensional stars array
            stars[current_star * 3]     = x;
            stars[current_star * 3 + 1] = y;
            stars[current_star * 3 + 2] = z;
            
            current_star++;
        }

        cout << endl;
        cout << "file: '" << star_files.at(j) << "'" << endl;
        cout << "    n_stars: " << setw(10) << count << endl;
        cout << endl;
        
        star_stream.close();
    }
    
    /** End storing stars ****************************************************/
    
    run_simulation();
    
    delete[] stars;
    delete[] k_means;
    delete[] cluster_sizes;
    delete[] assignments;

    return 0;
}
