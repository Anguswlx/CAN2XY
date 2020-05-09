//  Two-dimensional XY (planar spin) Model
//  J. Tobochnik and G.V. Chester, Phys. Rev. B 20, 3761 (1979)
//  S. Teitel and C. Jayaprakash, Phys. Rev. B 27, 598-601 (1983)
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
using namespace std;

inline double std_rand()
{
    return rand() / (RAND_MAX + 1.0);
}

int N = 16;                         // number of spins in x and y
double J_0 = 1.0;                   // spin-spin coupling constant
double f = 0.0;                     // uniform frustration constant
double T;                           // temperature in units of k_B/J_0

vector< vector<double> > theta;     // spin variable at each lattice site
vector< vector<double> > psi;       //

void initialize()
{
    theta.resize(N);
    psi.resize(2 * N);
    for (int i = 0; i < N; ++i)
        theta[i].resize(N);
    for (int i = 0; i < 2 * N; ++i)
        psi[i].resize(2 * N);
    if (f != 0.0) {
        // initialize psi
    }
}

double E(int x, int y)
{
    // energy of bonds connected to site (x,y)
    // find the nearest neighbor sites with periodic boundary conditions
    int x_plus = x + 1, x_minus = x - 1, y_plus = y + 1, y_minus = y - 1;
    if (x_plus  ==  N)  x_plus  = 0;
    if (x_minus == -1)  x_minus = N - 1;
    if (y_plus  ==  N)  y_plus  = 0;
    if (y_minus == -1)  y_minus = N - 1;

    // contribution to energy from 4 bonds at site (x,y)
    double E = -J_0 * (
        cos(theta[x][y] - theta[x_plus][y])  +
        cos(theta[x][y] - theta[x_minus][y]) +
        cos(theta[x][y] - theta[x][y_plus])  +
        cos(theta[x][y] - theta[x][y_minus]) );

    return E;
}

double E()
{
    // total energy
    double E_sum = 0;
    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            E_sum += E(x, y);
    return E_sum / 2;
}

bool metropolis_step(               // true/false if step accepted/rejected
    int x, int y,                   // change spin at size x,y
    double delta = 1.0)             // maximum step size in radians
{
    // find the local energy at site (x,y)
    double E_xy = E(x, y);

    // save the value of spin at (x,y)
    double theta_save = theta[x][y];

    // trial Metropolis step for spin at site (x,y)
    theta[x][y] += delta * (2 * std_rand() - 1);

    double E_xy_trial = E(x, y);

    // Metropolis test
    bool accepted = false;
    double dE = E_xy_trial - E_xy;
    double w = exp(- dE / T);
    if (w > std_rand())
        accepted = true;
    else
        theta[x][y] = theta_save;

    return accepted;
}

int monte_carlo_sweep()
{
    int acceptances = 0;
    // randomized sweep
    for (int step = 0; step < N * N; ++step) {
        // choose a site at random
        int x = int(N * std_rand());
        int y = int(N * std_rand());
        if (metropolis_step(x, y))
            ++acceptances;
    }
    return acceptances;
}

int main()
{
    cout << " Monte Carlo Simulation of the 2-dimensional XY Model\n"
         << " ----------------------------------------------------\n"
         << " " << N*N << " spins on " << N << "x" << N << " lattice,"
         << " uniform frustration f = " << f << endl;

    initialize();
//    for (int x = 0; x < N; ++x){
//        for (int y = 0; y < N; ++y){
// 	       cout << theta[x][y] << endl;
// 		}
// 	}
    T = 1/0.4;
	ofstream print("L16_beta0_4.dat");
    int equilibration_steps = 50000; // default 50000
    int production_steps = 50000000;

    cout << " performing " << equilibration_steps
         << " equilibration steps ..." << flush;
    for (int step = 0; step < equilibration_steps; ++step)
        monte_carlo_sweep();
    cout << " done" << endl;

    double e_av = 0, e_sqd_av = 0;      // average E per spin, squared average

    double accept_percent = 0;
    cout << " performing " << production_steps
         << " production steps ..." << flush;
    for (int step = 0; step < production_steps; ++step) {
        double num_spins = double(N) * double(N);
        accept_percent += monte_carlo_sweep() / num_spins;
        double e_step = E() / num_spins;
        e_av += e_step;
        e_sqd_av += e_step * e_step;
        if(step%10000==0){ //default 10000
		    for (int x = 0; x < N; ++x){
	            for (int y = 0; y < N; ++y){
				double theta_now=theta[x][y]/3.1415926535897;
				double cc=theta_now-floor(theta_now);
				theta_now=cc+int(floor(theta_now))%2;
				if(theta_now<0){theta_now = theta_now + 2;}
				print<< theta_now << '\t';
                }
	        }
		print << endl;
                cout << step/5000 << endl;        
       }
    }
    cout << " done" << endl;

    double per_step= 1.0 / double(production_steps);
    accept_percent *= per_step * 100.0;
    e_av *= per_step;
    e_sqd_av *= per_step;

    cout << " T = " << T << '\t'
         << " <e> = " << e_av << '\t'
         << " <e^2> = " << e_sqd_av << '\t'
         << " %accepts = " << accept_percent << endl;
    double c = (e_sqd_av - e_av * e_av) / (T * T);
    cout << " Heat capacity = " << c << endl;

    for (int x = 0; x < N; ++x){
	        for (int y = 0; y < N; ++y){
				double theta_now=theta[x][y]/3.1415926535897;
				double cc=theta_now-floor(theta_now);
				theta_now=cc+int(floor(theta_now))%2;
				if(theta_now<0){theta_now = theta_now + 2;}
				print<< theta_now << '\t';
     }
	}
    cout << cos(3.1415) << '\t' << cos(-46.2568) << '\t' << cos(180) << endl;
    return 0;
}
