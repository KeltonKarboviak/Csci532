#include <algorithm>
#include <unordered_map>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <stdio.h>

#include <mpi.h>


using std::vector;
using std::unordered_map;
using std::cout;
using std::cerr;
using std::ofstream;
using std::ifstream;
using std::endl;
using std::string;
using std::transform;
using std::ios;


#define READ_LENGTH 40


unordered_map<string, int> reads_map;

int comm_sz;  // Number of processes
int my_rank;  // Process rank


void usage(char *executable) {
    cerr << "Usage for the dna sequencer:" << endl;
    cerr << "\t" << executable << " <argument list>" << endl;
    cerr << "Required arguments:" << endl;
    cerr << "\t--seeds <str>*        : files containing the DNA reads" << endl;
    cerr << "\t--chrs <str>*  : files containing the DNA sequence to scan" << endl;
    cerr << "\t--output <str>        : file the output should be written to" << endl;
}

void getchar() {
    char dummy;
    cout << "Enter any key to continue." << endl;
    cin >> dummy;
}

int main(int argc, char **argv) {
    // Initialize the command-line arguments on every process
    MPI_Init(&argc, &argv);

    // Get the number of processes in MPI_COMM_WORLD, and put it in the
    // 'comm_sz' variable; ie., how many processes are running this program
    // (the same as what you put in the -np argument).
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    // Get the rank of this particular process in MPI_COMM_WORLD, and put it in
    // the 'my_rank' variable -- ie., what number is this process
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    vector<char*> seed_files, chromosome_files;
    char *output_filename;
    long *file_sizes = nullptr;
    long total_file_sizes = 0;

    // These arrays will hold the starting file # & position offsets within
    // the corresponding file for each process. The last index of both
    // arrays will be.
    long *file_start_offsets = new long[comm_sz];
    long *file_pos_offsets = new long[comm_sz];

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--seeds") == 0) {
            i++;
            while (i < argc && strlen(argv[i]) > 2 && !(argv[i][0] == '-' && argv[i][1] == '-')) {
                seed_files.push_back(argv[i++]);
            }
            i--;
        } else if (strcmp(argv[i], "--chrs") == 0) {
            i++;
            while (i < argc && strlen(argv[i]) > 2 && !(argv[i][0] == '-' && argv[i][1] == '-')) {
                chromosome_files.push_back(argv[i++]);
            }
            i--;
        } else if (strcmp(argv[i], "--output") == 0) {
            output_filename = argv[++i];
        } else {
            cerr << "Unknown argument '" << argv[i] << "'." << endl;
            usage(argv[0]);
        }
    }

    if (seed_files.size() <= 0) {
        cerr << "ERROR: seed file(s) not specified." << endl;
        usage(argv[0]);
    } else if (chromosome_files.size() <= 0) {
        cerr << "ERROR: chromosome file(s) not specified." << endl;
        usage(argv[0]);
    }

    file_sizes = new long[seed_files.size()];

    // Only Master process
    if (my_rank == 0) {
        // Get the file size for each seed file
        char* filename;
        long file_length;
        for (size_t i = 0, length = seed_files.size(); i < length; i++) {
            filename = seed_files.at(i);

            ifstream file_handle(filename);

            if (!file_handle.is_open()) {
                cerr << "Error opening file: " << filename << endl;
                exit(1);
            }

            // Seek to end of file, use tellg() to get size of entire file,
            // then store in array
            file_handle.seekg(0, file_handle.end);
            file_length = file_handle.tellg();
            file_sizes[i] = file_length;
            total_file_sizes += file_length;
        }

        // Calculate the file offsets and positions for each file
        int current_file = 0;
        long per_process = total_file_sizes / (comm_sz - 1);
        long left = per_process;
        int start_file = 0;
        int start_pos = 0;
        long current_file_sz = file_sizes[0];

        printf("per-process %ld\n", per_process);
        for (int i = 0, length = comm_sz - 1; i < length; i++) {
            printf("Calculating for process %d, start_file %d, start_pos %d\n", i + 1, start_file, start_pos);

            left = per_process;
            file_start_offsets[i] = start_file;
            file_pos_offsets[i] = start_pos;

            while (left > 0) {
                if (current_file_sz < left) {
                    left -= current_file_sz;
                    current_file_sz = file_sizes[++current_file];
                } else {
                    // This is end of the current process, need to setup for
                    // next process to be calculated
                    start_file = current_file;
                    start_pos = file_sizes[current_file] - (current_file_sz - left);
                    current_file_sz -= left;
                    left = 0;
                }

                printf("\tleft %ld, current_file_sz %ld\n", left, current_file_sz);
                getchar();
            }
        }

        file_start_offsets[comm_sz - 1] = seed_files.size() - 1;
        file_pos_offsets[comm_sz - 1] = file_sizes[file_start_offsets[comm_sz - 1]];
    }

    // Broadcast files_sizes out to all processes
//    MPI_Bcast(
//        file_sizes,        // the data we're broadcasting
//        seed_files.size(),  // the data size
//        MPI_LONG,           // the data type
//        0,                  // the process we're broadcasting from
//        MPI_COMM_WORLD
//    );

    // Broadcast file_start_offsets to all processes
    MPI_Bcast(
        file_start_offsets,  // the data we're broadcasting
        comm_sz,             // the data size
        MPI_LONG,            // the data type
        0,                   // the process we're broadcasting from
        MPI_COMM_WORLD
    );

    // Broadcast file_pos_offsets to all processes
    MPI_Bcast(
        file_pos_offsets,  // the data we're broadcasting
        comm_sz,           // the data size
        MPI_LONG,          // the data type
        0,                 // the process we're broadcasting from
        MPI_COMM_WORLD
    );

    /** Read in seed_file and insert into reads_map */
//    string line, read;
//    int hits;
//    while (getline(seed_file, line)) {
//        read = line.substr(0, READ_LENGTH);  // Get READ_LENGTH length read
//        hits = stoi(line.substr(READ_LENGTH + 1, string::npos));  // Get number of hits for the read
//        reads_map.insert({read, hits});
//    }
//    seed_file.close();
//
//    /** Read in genome_file into genome_str */
//    string genome_str = "";
//    getline(genome_file, line);  // Trash first line, which is a comment
//    while (getline(genome_file, line)) {
//        transform(line.begin(), line.end(), line.begin(), ::toupper);  // Convert string to all uppercase
//        genome_str.append(line);
//    }
//    genome_file.close();


    delete[] file_sizes;
    delete[] file_start_offsets;
    delete[] file_pos_offsets;


    MPI_Finalize();

    return 0;
}
