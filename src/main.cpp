#include "nelder_mead.h"
#include <Eigen/Dense>

using namespace std;

int main(){
    Nelder_Mead Solver;
    double* para = new double[1];
    para[0] = 1;
    Solver.solve(para, 1);
    return 0;
}
