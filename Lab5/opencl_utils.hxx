#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <fstream>
#include <vector>


/* check for an error and print an error message if found */
void check_error(cl_int err, const char* fmt, ...);

/* Find a GPU or CPU associated with the first available platform */
cl_device_id create_device();

/* Helper function to replace all occurrences of `original` with `replacement` within `str` */
void replace_all(std::string &str, const std::string original, const std::string replacement);

/* Helper function to load the Kernel program from `filename` into a std::string */
std::string load_kernel_as_string(const char* filename);

/* Create program from a file and compile it */
cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename, std::vector<std::string> replacements=std::vector<std::string>());
