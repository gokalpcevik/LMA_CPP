#include <Eigen/Dense>
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

// Single-precision floating point vector/matrix typedefs
template <size_t NumCoeff> using Vectorf = Eigen::Vector<float, NumCoeff>;
template <size_t NumRows, size_t NumCols> using Matrixf = Eigen::Matrix<float, NumRows, NumCols>;

struct SDatapoint
{
  // In seconds
  float Timestamp;
  // Fluorescence value (ADC)
  float Fluorescence;
};

// Compute Jacobian for the exponential curve
Eigen::Matrix<float, NUM_DATAPOINTS, COEFF_DOF> ComputeJr(Eigen::Vector<float, COEFF_DOF> const& coeffs,
                                                          std::array<SDatapoint, NUM_DATAPOINTS> const& dataset);
// Parse data from a csv file
bool ParseCSV(wchar_t const* file, TDataset& out_data);

// Solver Class
template <size_t NumCoeff, size_t NumDatapoints, bool Verbose> class CLevenbergMarquardtSolver
{
public:
  // Optimization Parameters
  Vectorf<NumCoeff> Beta = Vectorf<NumCoeff>::Zero();

  // NOTE: Ideally, we'd use Automatic Differentiation to compute this and not hand-derive the analytical Jacobian.
  // However, in embedded systems AD is computationally heavy.
  // Pointer to the function that computes the Jacobian of the residuals (user-supplied).
  Matrixf<NumDatapoints, NumCoeff> (*pf_Jr)(Vectorf<NumCoeff> const&,
                                            std::array<SDatapoint, NumDatapoints> const&) = nullptr;

  // NOTE: (1)
  // Currently using Lambda scaling by 10 as proposed by Levenberg-Marquardt in the original paper. Other alternatives:
  // 1. Adaptive Scaling Based on Trust Region
  // 2. Nielsen's Method
  // 3. Automatic Scale Detection
  // 4. Adaptive Restart
  // 5. Geodesic Acceleration
  // NOTE: (2):
  // For our case (decaying fluorescence exponential), scale-by-10 seems to work okay.
  void Solve(std::array<SDatapoint, NumDatapoints> const& dataset, size_t MaxIterations = 50, float Tol = 1e-6f,
             float Lambda0 = 1.0f)
  {
    float Lambda = Lambda0;
    float prev_error = std::numeric_limits<float>::max();

    size_t iteration = 0;
    for (iteration = 0; iteration < MaxIterations; ++iteration)
    {
      // Compute the Jacobian of the fitted function
      Matrixf<NumDatapoints, NumCoeff> Jr = this->pf_Jr(Beta, dataset);
      // Jacobian transpose
      Matrixf<NumCoeff, NumDatapoints> JrT = Jr.transpose();
      Vectorf<NumDatapoints> r = ComputeResiduals(dataset, Beta);
      // Compute current error
      float current_error = r.squaredNorm();
      printf("Current Error: %.3f\n", current_error);

      // Gramian of Jr
      Matrixf<NumCoeff, NumCoeff> G = JrT * Jr;
      // LMA damping factor
      Matrixf<NumCoeff, NumCoeff> damping_factor = Lambda * Matrixf<NumCoeff, NumCoeff>::Identity();
      Matrixf<NumCoeff, NumCoeff> H = G + damping_factor;
      // Solve step
      Vectorf<COEFF_DOF> Delta = H.colPivHouseholderQr().solve(JrT * r);

      // Check if the update step magnitude (norm) is bigger than the minimum tolerance
      if (Delta.norm() < Tol)
      {
        printf("Update step magnitude is lower than the allowed tolerance. Fitting has concluded.\n");
        break;
      }

      Vectorf<NumCoeff> new_beta = Beta + Delta;
      Vectorf<NumDatapoints> r_with_updated_beta = ComputeResiduals(dataset, new_beta);
      float new_error = r_with_updated_beta.squaredNorm();

      if (new_error < current_error)
      {
        Beta = new_beta;
        Lambda = std::max(Lambda / 10.0f, 1e-7f);
        prev_error = current_error;

        if constexpr (Verbose)
        {
          std::printf("Updated parameters yield smaller error.\n");
          std::printf("Decreasing Lambda by a factor of 10.\n");
        }
      }
      else
      {
        Lambda = std::min(Lambda * 10.0f, 1e7f);
        if constexpr (Verbose)
        {
          std::printf("Updated parameters yield larger error.\n");
          std::printf("Increasing Lambda by a factor of 10.\n");
        }
      }
    }
    if constexpr (Verbose)
    {
      printf("Final Lambda: %.3f\n", Lambda);
    }
    printf("MSE: %.3f\n", prev_error);
    printf("Total iterations: %zu\n", iteration);
  }

  Vectorf<NumDatapoints> ComputeResiduals(std::array<SDatapoint, NumDatapoints> const& dataset,
                                          Vectorf<NumCoeff> const& Params)
  {
    // Create timestamp vector using Map with stride
    Eigen::Map<const Vectorf<NumDatapoints>, Eigen::Unaligned, Eigen::InnerStride<2>> t(&dataset[0].Timestamp);
    // Create fluorescence vector using Map with stride
    Eigen::Map<const Vectorf<NumDatapoints>, Eigen::Unaligned, Eigen::InnerStride<2>> fluo(&dataset[0].Fluorescence);
    Vectorf<NumDatapoints> exp_term = Params[1] * (Params[2] * t).array().exp();
    // Compute f vector
    Vectorf<NumDatapoints> f = Params[0] * Eigen::Vector<float, NumDatapoints>::Ones() + exp_term;
    // Compute and return residuals
    return Vectorf<NumDatapoints>(fluo - f);
  }
};

// Program entry
int main(int argc, char** argv)
{
  TDataset dataset;
  if (!ParseCSV(L"m2.txt", dataset))
  {
    return EXIT_FAILURE;
  }

  CLevenbergMarquardtSolver<COEFF_DOF, NUM_DATAPOINTS, true> solver;
  // Set the Jacobian function
  solver.pf_Jr = ComputeJr;
  solver.Beta = {-20.0f, 100.0, -.06f};
  solver.Solve(dataset, 1000);

  printf("Parameters=[%.3f,%.3f,%.3f]\n", solver.Beta[0], solver.Beta[1], solver.Beta[2]);
  printf("Calculated Lifetime=%.3f\n", -1.0f / solver.Beta[2]);

  return EXIT_SUCCESS;
}

Matrixf<NUM_DATAPOINTS, COEFF_DOF> ComputeJr(Vectorf<COEFF_DOF> const& coeffs,
                                             std::array<SDatapoint, NUM_DATAPOINTS> const& dataset)
{
  Matrixf<NUM_DATAPOINTS, COEFF_DOF> Jr = Matrixf<NUM_DATAPOINTS, COEFF_DOF>::Zero();
  float Beta0 = coeffs[0];
  float Beta1 = coeffs[1];
  float Beta2 = coeffs[2];
  for (size_t rowidx = 0; rowidx < NUM_DATAPOINTS; ++rowidx)
  {
    float t = dataset[rowidx].Timestamp;
    Jr.row(rowidx) << 1.0f, std::exp(Beta2 * t), Beta1 * t * std::exp(Beta2 * t);
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
    // std::printf("Fluorescence [ADC]: %.1f, Timestamp (us): %.3f\r\n", fluo, timestamp);

    // Set the data
    data.Fluorescence = fluo;
    data.Timestamp = timestamp;
    ++index;
  }
  return true;
}
