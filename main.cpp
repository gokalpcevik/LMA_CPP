#include <Eigen/Eigen>
#include <fstream>
#include <iostream>

// We want to fit y(t)=A+B*exp(C*t), where t is time, {A,B,C} are the fit coefficients
static constexpr size_t COEFF_DOF = 3;
// Number of datapoints (points in the decay curve)
static constexpr size_t NUM_DATAPOINTS = 80;

// Eigen IO formatter for printing vectors and matrices
Eigen::IOFormat OctaveFmt(Eigen::StreamPrecision, 0, ", ", ";\n", "", "", "[", "]");

struct SDatapoint;
// Dataset typedef
using TDataset = std::array<SDatapoint, NUM_DATAPOINTS>;
// Coefficients
Eigen::Vector<float, COEFF_DOF> g_coeff;

struct SDatapoint
{
  // In seconds
  float timestamp;
  // Fluorescence value (ADC)
  float fluo;
};

// Compute Jacobian for the exponential curve
Eigen::Matrix<float, NUM_DATAPOINTS, COEFF_DOF> ComputeJr(Eigen::Vector<float, COEFF_DOF> const& coeffs,
                                                          std::array<SDatapoint, NUM_DATAPOINTS> const& dataset);
// Parse data from a csv file
bool ParseCSV(wchar_t const* file, TDataset& out_data);

template <size_t NumCoeff, size_t NumDatapoints, bool Verbose> class CLevenbergMarquardtSolver
{
public:
  void Solve(std::array<SDatapoint, NumDatapoints> const& dataset, size_t MaxIterations = 50, float Tol = 1e-6f,
             float Lambda0 = 1.0f)
  {
    // TODO: Implement
  }

  // Residual vector
  Eigen::Vector<float, NumDatapoints> r = Eigen::Vector<float, NumDatapoints>::Zero();
  // Sum of squared errors
  float MSE = 0.0f;
  // Optimization Parameters
  Eigen::Vector<float, NumCoeff> Beta = Eigen::Vector<float, NumCoeff>::Zero();
  // Update Step
  Eigen::Vector<float, NumCoeff> DeltaP = Eigen::Vector<float, NumCoeff>::Zero();
  // Levenberg-Marquardt damping parameter
  float Lambda = 1.0f;

  // Pointer to the function that computes the Jacobian of the residuals (user-supplied).
  // NOTE: Ideally, we'd use Automatic Differentiation to compute this and not hand-derive the analytical Jacobian.
  // However, in embedded systems AD is computationally heavy.
  Eigen::Matrix<float, NumDatapoints, NumCoeff> (*Jr)(Eigen::Vector<float, NumCoeff> const&,
                                                      std::array<SDatapoint, NumDatapoints> const&) = nullptr;
};

// Program entry
int main(int argc, char** argv)
{
  CLevenbergMarquardtSolver<COEFF_DOF, NUM_DATAPOINTS> solver;
  // Set the Jacobian function
  solver.Jr = ComputeJr;

  TDataset dataset;
  ParseCSV(L"m1.txt", dataset);

  return EXIT_SUCCESS;
}

Eigen::Matrix<float, NUM_DATAPOINTS, COEFF_DOF> ComputeJr(Eigen::Vector<float, COEFF_DOF> const& coeffs,
                                                          std::array<SDatapoint, NUM_DATAPOINTS> const& dataset)
{
  Eigen::Matrix<float, NUM_DATAPOINTS, COEFF_DOF> Jr = Eigen::Matrix<float, NUM_DATAPOINTS, COEFF_DOF>::Zero();
  float Beta0 = coeffs[0];
  float Beta1 = coeffs[1];
  float Beta2 = coeffs[2];
  for (size_t rowidx = 0; rowidx < NUM_DATAPOINTS; ++rowidx)
  {
    float t = dataset[rowidx].timestamp;
    Jr.row(rowidx) << -1.0f, -std::exp(Beta2 * t), -Beta1 * t * std::exp(Beta2 * t);
  }
  return std::move(Jr);
}

bool ParseCSV(wchar_t const* file, TDataset& out_data)
{
  std::ifstream handle;
  handle.open(file, std::ios::in);

  if (!handle.is_open())
  {
    std::printf("Cannot open file %ls\r\n", file);
    return false;
  }
  std::string line;
  size_t index = 0;
  while (std::getline(handle, line))
  {
    SDatapoint& data = out_data[index];
    size_t loc_comma = line.find(",");
    size_t loc_eol = line.find("\n");
    // Parse the fluorescence and timestamp values
    float fluo = std::stof(line.substr(0, loc_comma));
    float timestamp = std::stof(line.substr(loc_comma + 1, loc_eol - loc_comma - 1));
    std::printf("Fluorescence [ADC]: %.1f, Timestamp (us): %.3f\r\n", fluo, timestamp);

    // Set the data
    data.fluo = fluo;
    data.timestamp = timestamp;
    ++index;
  }
  return true;
}
