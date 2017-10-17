#include <iomanip>
using std::fixed;
using std::setprecision;
using std::setw;

#include <iostream>
using std::cout;
using std::endl;

#include <fstream>
using std::ofstream;

#include <sstream>
using std::ostringstream;

#include <string>
using std::string;


//write the state of the smulation to a file
void write_simulation_state(string name, int height, int width, int time_step, double **values) {
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


int main(int argc, char **argv) {
    int height, width;
    int vertical_slices, horizontal_slices;
    int time_steps;

    if (argc != 7) {
        cout << "ERROR, incorrect arguments." << endl;
        cout << "usage:" << endl;
        cout << "\t" << argv[0] << " <simulation name : string> <height : int> <width : int> <vertical slices : int> <horizontal slices : int> <time steps : int>" << endl;
        exit(1);
    }

    string simulation_name(argv[1]);

    //the height and width of the simulation
    height = atoi(argv[2]);
    width = atoi(argv[3]);

    //horizontal and vertical slices will be used to determine how to divide up your MPI processes
    //You will need to have a number of MPI processes equal to vertical_slices * horizontal_slices
    vertical_slices = atoi(argv[4]);
    horizontal_slices = atoi(argv[5]);

    //how long to run the simulation for
    time_steps = atoi(argv[6]);

    //initialize all values to 0
    double **values = new double*[height];
    double **values_next = new double*[height];
    for (uint32_t i = 0; i < height; i++) {
        values[i] = new double[width];
        values_next[i] = new double[width];
        for (uint32_t j = 0; j < width; j++) {
            values[i][j] = 0.0;
            values_next[i][j] = 0.0;
        }
    }

    //put a heat source on the left column of the simulation
    for (uint32_t i = 0; i < height; i++) {
        values[i][0] = 1.0;
        values_next[i][0] = 1.0;
    }

    //put a cold source on the left column of the simulation
    for (uint32_t i = 0; i < height; i++) {
        values[i][width - 1] = -1.0;
        values_next[i][width - 1] = -1.0;
    }

    write_simulation_state(simulation_name, height, width, 0, values);

    //update the heat values at each step of the simulation for all internal values
    for (uint32_t time_step = 1; time_step <= time_steps; time_step++) {
        //the border values are sources/sinks of heat
        for (uint32_t i = 1; i < height - 1; i++) {
            for (uint32_t j = 1; j < width - 1; j++) {
                double up = values[i - 1][j];
                double down = values[i + 1][j];
                double left = values[i][j - 1];
                double right = values[i][j + 1];

                //set the values of the next time step of the heat simulation
                values_next[i][j] = (up + down + left + right) / 4.0;
            }
        }

        //swap the values arrays
        double **temp = values_next;
        values_next = values;
        values = temp;

        //store the simulation state so you can compare this to your MPI version
        write_simulation_state(simulation_name, height, width, time_step, values);
    }
}
