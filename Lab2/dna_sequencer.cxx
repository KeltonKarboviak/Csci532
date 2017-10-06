#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <stdio.h>
#include <mpi.h>

using std::cerr;
using std::cout;
using std::endl;
using std::ifstream;
using std::map;
using std::ofstream;
using std::string;
using std::transform;
using std::unordered_map;
using std::vector;


#define TERM_TAG    0
#define RESULT_TAG  1
#define READ_LENGTH 40


map<int, map<int, int> > master_map;
map<string, int> reads_map;

vector<char*> seed_files, chromosome_files;

char *output_filename;

unsigned long *file_sizes = nullptr;
long *file_start_offsets = nullptr;
unsigned long *file_pos_offsets = nullptr;

int comm_sz;  // Number of processes
int my_rank;  // Process rank


void output_map_results() {
    ofstream output_file(output_filename);

    for (auto iter1 = master_map.begin(); iter1 != master_map.end(); iter1++)
        for (auto iter2 = iter1->second.begin(); iter2 != iter1->second.end(); iter2++)
            output_file << iter1->first << ", " << iter2->first << ", " << iter2->second << endl;
}

/**
 * This function for the Master process will asynchronously receive results
 * from the Worker processes.  Once it has received an exit message from each
 * Worker process, then the Master will stop processing.
 */
void do_master_stuff() {
    printf("Master process starting\n");

    int num_finished = 0;
    int num_workers = comm_sz - 1;
    int result_array[3];

    MPI_Request request;
    MPI_Status status;

    int chr_file_idx, chr_match_idx, hits;

    while (num_finished < num_workers) {
        // Receive an asynchronous message from any source with any tag
        MPI_Irecv(
            result_array,
            3,
            MPI_INT,
            MPI_ANY_SOURCE,
            MPI_ANY_TAG,
            MPI_COMM_WORLD,
            &request
        );

        // Block until it gets here
        MPI_Wait(&request, &status);

        if (status.MPI_TAG == RESULT_TAG) {  // Result message
            // Grab all of the components from the array, then insert into
            // the master_map
            chr_file_idx = result_array[0];
            chr_match_idx = result_array[1];
            hits = result_array[2];

            master_map[chr_file_idx][chr_match_idx] += hits;
        } else {                             // Termination message
            num_finished++;
        }
    }

    // Output all the results we've received to the output file
    output_map_results();

    printf("Master finished\n");
}

/**
 * This function sends an asynchronous result message to the Master process
 * through MPI.
 *
 * @param current_file int index of the chromosome file in which the match
 *                     occurred
 * @param chr_idx      int index within the chromosome in which the match
 *                     occurred
 * @param hits         int number of times the read occurs in the read map
 */
void send_match_msg(int current_file, int chr_idx, int hits) {
    // Fill the result_array for sending, result is in the form
    // [ <file index>, <index matched in chr>, <number of hits for the read> ]
    int result_array[3] = {current_file, chr_idx, hits};

    MPI_Request request;
    // Asynchronously send the result to the Master process
    MPI_Isend(
        result_array,  // the data we're sending
        3,             // the number of elements
        MPI_INT,       // the data type
        0,             // the target process
        RESULT_TAG,    // the message tag
        MPI_COMM_WORLD,
        &request       // an MPI_Request for tracking when the send is done
    );
}

/**
 * This function sends an asynchronous termination message to the Master
 * process through MPI.
 */
void send_term_msg() {
    // Fill the result_array with dummy values, since all that matters is the
    // special TERM_TAG that we're sending
    int result_array[3] = {0, 0, 0};

    MPI_Request request;
    // Asynchronously send the result to the Master process
    MPI_Isend(
        result_array,  // the data we're sending
        3,             // the number of elements
        MPI_INT,       // the data type
        0,             // the target process
        TERM_TAG,    // the message tag
        MPI_COMM_WORLD,
        &request       // an MPI_Request for tracking when the send is done
    );
}

void check_for_matches(int current_file, string chr_str) {
    string compare;

    int i, length = chr_str.length() - READ_LENGTH + 1;
    for (i = 0; i < length; i++) {
        compare = chr_str.substr(i, READ_LENGTH);

        // The compare string doesn't contain any N's, and it matches something
        // in the reads_map
        if (compare.find('N') == string::npos && reads_map.count(compare) > 0) {
            send_match_msg(current_file, i, reads_map[compare]);
        }
    }
}

/**
 * This function for Worker processes will first read in their respective chunks
 * of the seed file(s), then they will scan the chromosome file(s) and attempt
 * to find matches.
 */
void do_worker_stuff() {
    // Need to subtract 1 from the rank for accessing offset arrays since the
    // Master process does not access the offset arrays while processing
    int my_idx = my_rank - 1;

    printf("Worker %d: Reading seeds, file %ld pos %ld\n", my_idx, file_start_offsets[my_idx], file_pos_offsets[my_idx]);

    /** Read in seed_file and insert into reads_map */
    string line;
    char current_char;
    unsigned long current_pos, end_pos;
    int current_file = file_start_offsets[my_idx];
    int end_file = file_start_offsets[my_idx + 1];
    for (; current_file <= end_file; current_file++) {
        ifstream seed_file(seed_files.at(current_file));

        current_pos = current_file == file_start_offsets[my_idx]
            ? file_pos_offsets[my_idx]
            : 0;

        end_pos = current_file != end_file
            ? file_sizes[current_file]
            : file_pos_offsets[my_idx + 1];

        printf("\tWorker %d: current_file %d, end_file %d, current_pos %ld, end_pos %ld\n", my_idx, current_file, end_file, current_pos, end_pos);

        /**
         * Find initial position in the file since this process's starting position may be in the middle of a line
         */
        // Seek to starting position
        seed_file.seekg(current_pos);

        // If our current_pos is at the end of the file, then continue onto next file
        if (!seed_file.get(current_char))
            continue;

        while (true) {
            if (current_char == '@') {
                printf("\tWorker %d: pos %ld @\n", my_idx, current_pos);
                getline(seed_file, line);  // Gets everything on the line except beginning @, this is trash to us
                break;
            }

            if (current_char == '\n') {
                printf("\tWorker %d: pos %ld \\n\n", my_idx, current_pos);
                // We've hit a new line, so now travel forwards until we find an @
                do {
                    getline(seed_file, line);  // getline() after a \n will give us the entire next line
                } while (line[0] != '@');

                current_pos = seed_file.tellg();

                // break because we've found our 'actual' starting position
                break;
            }

            printf("\tWorker %d: pos %ld\n", my_idx, current_pos);

            // We're going to travel backwards until we find an @ or \n
            current_pos--;
            seed_file.seekg(current_pos);
            seed_file.get(current_char);
        }

        printf("\t\tWorker %d: actual_pos %ld\n", my_idx, current_pos);

        /**
         * Loop through the rest of this process's chunk and insert the reads into its map
         */
        while (current_pos < end_pos) {
            getline(seed_file, line);

            // Only insert reads into reads_map if it doesn't contain any N's
            if (line.find('N') == string::npos)
                reads_map[line]++;

            // Trash next 3 lines
            getline(seed_file, line);
            getline(seed_file, line);
            getline(seed_file, line);

            current_pos = seed_file.tellg();
        }

        seed_file.close();
    }

    for (auto it = reads_map.begin(); it != reads_map.end(); it++) {
        cout << my_idx << " " << it->first << " " << it->second << endl;
    }

    printf("Worker %d: Reading Genome\n", my_idx);

    /**
     * Go through all chromosome files
     */
    string chr_str;
    for (size_t i = 0; i < chromosome_files.size(); i++) {
        ifstream chr_file(chromosome_files.at(i));

        chr_str = "";
        getline(chr_file, line);  // Trash first line, which is a comment
        while (getline(chr_file, line)) {
            transform(line.begin(), line.end(), line.begin(), ::toupper);  // Convert string to all uppercase
            chr_str.append(line);
        }

        // Check the entire chromosome string for matches in the reads_map
        check_for_matches(i, chr_str);

        chr_file.close();
    }

    // Worker is done so send termination message
    send_term_msg();

    printf("Worker %d: Finished\n", my_idx);
}

void usage(char *executable) {
    cerr << "Usage for the dna sequencer:" << endl
         << "\t" << executable << " <argument list>" << endl
         << "Required arguments:" << endl
         << "\t--seeds <str>*        : files containing the DNA reads" << endl
         << "\t--chrs <str>*  : files containing the DNA sequence to scan" << endl
         << "\t--output <str>        : file the output should be written to" << endl
         << "** Note: this requires >= 2 processes to run properly **" << endl;

    exit(1);
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

    // Make sure there is more than 1 process running
    if (comm_sz < 2)
        usage(argv[0]);

    unsigned long total_file_sizes = 0;

    // These arrays will hold the starting file # & position offsets within
    // the corresponding file for each process. The last index of both
    // arrays will be.
    file_start_offsets = new long[comm_sz];
    file_pos_offsets = new unsigned long[comm_sz];

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

    if (my_rank == 0) {
        cout << "Parsed command-line args:" << endl
             << "\tNumber of seed files:       " << seed_files.size() << endl
             << "\tNumber of chromosome files: " << chromosome_files.size() << endl
             << "\tOutput filename:            " << output_filename << endl;
    }

    file_sizes = new unsigned long[seed_files.size()];

    // Only Master process
    if (my_rank == 0) {
        // Get the file size for each seed file
        char* filename;
        unsigned long file_length;
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
        unsigned long per_process = total_file_sizes / (comm_sz - 1);
        unsigned long left = per_process;
        int start_file = 0;
        unsigned long start_pos = 0;
        unsigned long current_file_sz = file_sizes[0];

        printf("per-process %ld\n", per_process);
        for (int i = 0, length = comm_sz - 1; i < length; i++) {
            printf("Calculating for Worker %d, start_file %d, start_pos %ld\n", i, start_file, start_pos);

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

        printf("Calculation results:\n");
        for (int i = 0; i < comm_sz; i++) {
            printf("\t%d: start_file %ld, start_pos %ld\n", i, file_start_offsets[i], file_pos_offsets[i]);
        }
        printf("\n");
    }

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
        file_pos_offsets,   // the data we're broadcasting
        comm_sz,            // the data size
        MPI_UNSIGNED_LONG,  // the data type
        0,                  // the process we're broadcasting from
        MPI_COMM_WORLD
    );

    if (my_rank == 0) {  // Only Master process
        do_master_stuff();
    } else {             // Only Worker processes
        do_worker_stuff();
    }

    delete[] file_sizes;
    delete[] file_start_offsets;
    delete[] file_pos_offsets;

    MPI_Finalize();

    return 0;
}
