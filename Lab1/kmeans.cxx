/* W B Langdon at MUN 10 May 2007
 * Program to demonstarte use of OpenGL's glDrawPixels
 */

/*#ifdef _WIN32
#include <windows.h>
#endif

#ifdef __APPLE__
#  include <OpenGL/gl.h>
#  include <OpenGL/glu.h>
#  include <GLUT/glut.h>
#else
#  include <GL/gl.h>
#  include <GL/glu.h>
#  include <GL/glut.h>
#endif*/

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

/*int window_size;
int window_width;
int window_height;

int     ox                  = 0;
int     oy                  = 0;
int     buttonState         = 0; 
float   camera_trans[]      = {0, -0.2, -10};
float   camera_rot[]        = {0, 0, 0};
float   camera_trans_lag[]  = {0, -0.2, -10};
float   camera_rot_lag[]    = {0, 0, 0};
const float inertia         = 0.1f;*/


/*void reshape(int w, int h) {
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (float) w / (float) h, 0.1, 1000.0);

    glMatrixMode(GL_MODELVIEW);
    glViewport(0, 0, w, h);
}

void mouse_button(int button, int state, int x, int y) {
    int mods;

    if (state == GLUT_DOWN)
        buttonState |= 1<<button;
    else if (state == GLUT_UP)
        buttonState = 0;

    mods = glutGetModifiers();
    if (mods & GLUT_ACTIVE_SHIFT) 
    {
        buttonState = 2;
    } 
    else if (mods & GLUT_ACTIVE_CTRL) 
    {
        buttonState = 3;
    }

    ox = x; oy = y;

    glutPostRedisplay();
}

void mouse_motion(int x, int y) {
    float dx = (float)(x - ox);
    float dy = (float)(y - oy);

    if (buttonState == 3) 
    {
        // left+middle = zoom
        camera_trans[2] += (dy / 100.0f) * 0.5f * fabs(camera_trans[2]);
    } 
    else if (buttonState & 2) 
    {
        // middle = translate
        camera_trans[0] += dx / 100.0f;
        camera_trans[1] -= dy / 100.0f;
    }
    else if (buttonState & 1) 
    {
        // left = rotate
        camera_rot[0] += dy / 5.0f;
        camera_rot[1] += dx / 5.0f;
    }

    ox = x; oy = y;
    glutPostRedisplay();
}


/**
 *  The display function gets called repeatedly, updating the visualization of the simulation
 */
/*void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    cout << "trans: " << camera_trans[0] << ", " << camera_trans[1] << ", " << camera_trans[2] << " -- rot: " << camera_rot[0] << ", " << camera_rot[1] << ", " << camera_rot[2] << endl;
    for (int c = 0; c < 3; ++c)
    {
        camera_trans_lag[c] += (camera_trans[c] - camera_trans_lag[c]) * inertia;
        camera_rot_lag[c] += (camera_rot[c] - camera_rot_lag[c]) * inertia;
    }

    glTranslatef(camera_trans_lag[0], camera_trans_lag[1], camera_trans_lag[2]);
    glRotatef(camera_rot_lag[0], 1.0, 0.0, 0.0);
    glRotatef(camera_rot_lag[1], 0.0, 1.0, 0.0);

    glBegin(GL_POINTS);
    int count = 0;
    float red = 1.0, green = 1.0, blue = 1.0;

    for (int j = 0; j < n_files; j++) {

        glColor3f(red, green, blue);

        for (int i = 0; i < n_stars[j]; i++) {
            if ((count % modulo) == 0) glVertex3f(stars[j][i][0], stars[j][i][1], stars[j][i][2]);
            count++;
        }

        // Note this only works for up to 27 different input files.
        red -= 0.30;
        if (red < 0) {
            red = 1;
            green -= 0.30;

            if (green < 0) {
                green = 1;
                blue -= 0.3;
            }
        }
    }
    glEnd();

    glFlush();
    glutSwapBuffers();

    glutPostRedisplay();
}*/


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
    for (int i = 0; i < star_files.size(); i++) {
        cout << "        '" << star_files.at(i) << "'" << endl;
    }
    cout << endl;
    
    /** End command-line args & star count ***********************************/

    /** Open each file and store all stars into a single one-dimensional array */

    // Allocate array, we are going to put all the stars coordinates into a
    // one-dimenisonal array
    stars = new double[n_stars * 3];
    
    int current_star = 0;
    for (int j = 0; j < n_files; j++) {
        ifstream star_stream(star_files.at(j).c_str());

        // Get number of stars in current file
        star_stream >> count;

        cout << "Reading " << count << " stars." << endl;

        double l, b, r;
        for (int i = 0; i < count; i++) {
            star_stream >> l >> b >> r;

            // convert degrees to radians
            l = l * M_PI / 180;
            b = b * M_PI / 180;

            // convert l b r (galactic) to x y z (cartesian)
            stars[current_star * 3]     = r * cos(b) * sin(l);
            stars[current_star * 3 + 1] = 4.2 - r * cos(l) * cos(b);
            stars[current_star * 3 + 2] = r * sin(b);
            
            current_star++;
        }

        cout << endl;
        cout << "file: '" << star_files.at(j).c_str() << "'" << endl;
        cout << "    n_stars: " << setw(10) << count << endl;
        cout << endl;
        
        star_stream.close();
    }
    
    /** End storing stars ****************************************************/

    return 0;
}
