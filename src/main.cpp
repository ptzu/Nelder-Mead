#include "nelder_mead.h"
#include <Eigen/Dense>

using namespace std;

int main(){
    Nelder_Mead Solver;
    int dim = 2;
    double* para = new double[dim];
    para[0] = 100; para[1] = 100;
    cout << "Before optimization: " << para[0] << " " << para[1] << endl;
    Solver.solve(para, dim);
    cout << "After optimization: " << para[0] << " " << para[1] << endl;
    return 0;
}
