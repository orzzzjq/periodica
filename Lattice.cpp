#include <iostream>
#include <vector>
#include <Eigen/Dense>

using namespace Eigen;

template <typename integer>
class Lattice {
	typedef Matrix<integer, Dynamic, Dynamic> Matrix;

private:
	Matrix V; // basis of the lattice

public:
	Lattice(int dimension) { V = Matrix(dimension, 0); }
	Lattice(int dimension, int size) { V = Matrix(dimension, size); }
	Lattice(const Matrix& V) : V(V) {}

	integer dimension() const { return V.rows(); }
	integer size() const { return V.cols(); }
	Matrix& basis() { return V; }

	integer operator()(int i, int j) const { return V(i, j); }
	integer& operator()(int i, int j) { return V(i, j); }

	// joint two lattices
	void operator+=(const Lattice& L) {
		if (this->dimension() != L.dimension()) {
			throw std::invalid_argument("lattices in different dimensions");
		}
		Matrix NewBasis(this->dimension(), this->size() + L.size());
		for (int i = 0; i < this->dimension(); ++i) {
			for (int j = 0; j < this->size(); ++j) {
				NewBasis(i, j) = V(i, j);
			}
			for (int j = 0; j < L.size(); ++j) {
				NewBasis(i, j + this->size()) = L(i, j);
			}
		}
		V = HermiteNormalForm(NewBasis);
	}

	// compute HNF of an integer matrix
	Matrix HermiteNormalForm(Matrix& M) {
		// column index, for implicit column exchange
		std::vector<int> colid(M.cols()); 
		for (int j = 0; j < M.cols(); ++j) colid[j] = j;
		
		// function for exchanging two column indices
		auto exchange = [](int& x, int& y) { int z = x; x = y, y = z; };

		// the reduction algorithm
		int rows = M.rows(), cols = M.cols();
		int i, j = 0, k, l;
		for (i = 0; i < rows; ++i) {
			// find the first l >= j such that M(i, l) != 0
			bool found = 0;
			for (l = j; l < cols; ++l) {
				if (M(i, colid[l]) != 0) {
					exchange(colid[j], colid[l]);
					found = 1;
					break;
				}
			}
			if (!found) continue;
			// make sure M(i, j) > 0
			if (M(i, colid[j]) < 0) {
				for (l = i; l < rows; ++l) {
					M(l, colid[j]) = -M(l, colid[j]);
				}
			}
			for (k = j + 1; k < cols; ++k) {
				if (M(i, colid[k]) != 0) {
					// make sure M(i, k) > 0
					if (M(i, colid[k]) < 0) {
						for (l = i; l < rows; ++l) {
							M(l, colid[k]) = -M(l, colid[k]);
						}
					}
					// euclidean algorithm for gcd
					while (M(i, colid[j]) > 0 && M(i, colid[k]) > 0) {
						integer t = M(i, colid[j]) / M(i, colid[k]);
						if (t) { // substrace t * M(:, k) from M(:, j)
							for (l = i; l < rows; ++l) {
								M(l, colid[j]) -= M(l, colid[k]) * t;
							}
						}
						exchange(colid[j], colid[k]);
					}
				}
			}
			// reduce columns precede to j
			for (k = 0; k < j; ++k) {
				integer t = M(i, colid[k]) / M(i, colid[j]);
				if (t) { // substrace t * M(:, j) from M(:, k)
					for (l = i; l < rows; ++l) {
						M(l, colid[k]) -= M(l, colid[j]) * t;
					}
				}
			}
			j += 1;
		}
		
		// collect the final result
		int finalsize = j;
		Matrix Reduced(rows, finalsize);
		for (i = 0; i < rows; ++i) {
			for (j = 0; j < finalsize; ++j) {
				Reduced(i, j) = M(i, colid[j]);
			}
		}
		return Reduced;
	}
};

using namespace std;

int main()
{
	typedef int integer;

	Lattice<integer> L(2);
	cout << L.basis() << endl;

	Lattice<integer> L1(2, 1), L2(2, 1);
	
	L1(0, 0) = 2;
	L1(1, 0) = 0;

	L2(0, 0) = 3;
	L2(1, 0) = 0;

	L += L2;
	cout << L.basis() << endl;
	L += L1;
	cout << L.basis() << endl;
}
