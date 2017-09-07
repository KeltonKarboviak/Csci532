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

#include <cstring>
#include <string>
#include <queue>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <iomanip>
#include "math.h"

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

int modulo = 1;
int *n_stars = NULL;
int n_files = -1;
double ***stars = NULL;

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
    cerr << "Usage for star visualizer:" << endl;
    cerr << "    " << executable << " <argument list>" << endl;
    cerr << "Possible arguments:" << endl;
    cerr << "   --window_size <width> <height>  : width and height of the window <int> <int>" << endl;
    cerr << "   --star_files <str>*             : files containing the stars (in LBR coordinates) followed by a cluster identifier (space separated)" << endl;
    cerr << "   --modulo <int>                  : draw every <modulo> stars, default 1" << endl;
    exit(1);
}

int main(int argc, char** argv) {

    vector<string> star_files;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--star_files") == 0) {
            i++;
            while (i < argc && strlen(argv[i]) > 2 && !(argv[i][0] == '-' && argv[i][1] == '-')) {
                star_files.push_back(string(argv[i]));
                i++;
            }
            i--;

        } else if (strcmp(argv[i], "--window_size") == 0) {
            window_width = atoi(argv[++i]);
            window_height = atoi(argv[++i]);

        } else if (strcmp(argv[i], "--modulo") == 0) {
            modulo = atoi(argv[++i]);


        } else {
            cerr << "Unknown argument '" << argv[i] << "'." << endl;
            usage(argv[0]);
        }
    }

    n_files = star_files.size();

    if (n_files == 0) {
        cerr << "ERROR: star file not specified." << endl;
        usage(argv[0]);
    }

    cout << "Arguments succesfully parsed." << endl;
    cout << "    star files:    " << endl;
    for (int i = 0; i < star_files.size(); i++) {
        cout << "        '" << star_files.at(i) << "'" << endl;
    }
    cout << "  window_width:    " << setw(10) << window_width << endl;
    cout << "  window_height:   " << setw(10) << window_width << endl;

    window_size = window_width * window_height;

    stars = new double**[n_files];
    n_stars = new int[n_files];

    for (int j = 0; j < n_files; j++) {
        ifstream star_stream(star_files.at(j).c_str());

        star_stream >> n_stars[j];

        if (n_stars[j] <= 0) {
            cerr << "Incorrectly formatted star file: '" << star_files.at(j).c_str() << "'" << endl;
            cerr << "First line should contain the number of stars in the file, and be > 0." << endl;
            exit(1);
        }

        stars[j] = new double*[n_stars[j]];

        cout << "Reading " << n_stars[j] << " stars." << endl;

        double avg_l = 0.0, avg_b = 0.0, avg_r = 0.0, avg_x = 0.0, avg_y = 0.0, avg_z = 0.0;
        double l, b, r;
        for (int i = 0; i < n_stars[j]; i++) {
            stars[j][i] = new double[4];

            star_stream >> l >> b >> r;

//            cout << "lbr:   " << setw(15) << l << setw(15) << b << setw(15) << r << endl;

            avg_l += l;
            avg_b += b;
            avg_r += r;

            // convert degrees to radians
            l = l * M_PI / 180;
            b = b * M_PI / 180;

    //        stars[i][0] = stars[i][0] * M_PI / 180;
    //        stars[i][1] = ((-1.0 * stars[i][1]) + 90) * M_PI / 180;

            //convert lbr (galactic) to x y z (cartesian)
            stars[j][i][0] = r * cos(b) * sin(l);
            stars[j][i][1] = 4.2 - r * cos(l) * cos(b);
            stars[j][i][2] = r * sin(b);

            stars[j][i][3] = j;
                
            avg_x += stars[j][i][0];
            avg_y += stars[j][i][1];
            avg_z += stars[j][i][2];

//            cout << "xyz:   " << setw(15) << stars[j][i][0] << setw(15) << stars[j][i][1] << setw(15) << stars[j][i][2] << endl;
//            cout << endl;

    //        if ((i % 1000) == 0) cout << "read " << i << " stars." << endl;
        }

        avg_l /= n_stars[j];
        avg_b /= n_stars[j];
        avg_r /= n_stars[j];
        avg_x /= n_stars[j];
        avg_y /= n_stars[j];
        avg_z /= n_stars[j];


        cout << endl;
        cout << "file: '" << star_files.at(j).c_str() << "'" << endl;
        cout << "   n_stars: " << setw(10) << n_stars[j] << endl;
        cout << "   avg l:   " << setw(10) << avg_l << endl;
        cout << "   avg b:   " << setw(10) << avg_b << endl;
        cout << "   avg r:   " << setw(10) << avg_r << endl;
        cout << "   avg x:   " << setw(10) << avg_x << endl;
        cout << "   avg y:   " << setw(10) << avg_y << endl;
        cout << "   avg z:   " << setw(10) << avg_z << endl;
        cout << endl;
    }

    cout << "Initialized star visualizer!" << endl;
    cout << "window width: "    << window_width << endl;
    cout << "window height: "   << window_height << endl;
    cout << "window size : "    << window_size << endl;

    /**
     *  Generate the first event -- start a fire at a random (non-water) cell
     */

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);

    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Star Visualization");

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutMouseFunc(mouse_button);
    glutMotionFunc(mouse_motion);
    //glutKeyboardFunc(keyboard);
    //glutIdleFunc(idle);

    glEnable(GL_DEPTH_TEST);
    glClearColor(0.0, 0.0, 0.0, 1.0);
//    glPointSize(2);

    glutMainLoop();
}
