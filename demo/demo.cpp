#include <iostream>
#include <vector>
#include <array>
#include <string>

#include "cnpy.h"


using namespace std;
using RGB = array<char, 3>;

#define Ly 1080
#define Lx 1920


vector<vector<RGB>> get_2d_space_rgb(string input_path) {

	cnpy::NpyArray raw_rgb = cnpy::npy_load(input_path);
	char *data = raw_rgb.data<char>();
	int pos = 0;

	vector<vector<RGB>> ly_vec;
	for (int j = 0; j < Ly; ++j) {
		vector<RGB> lx_vec;
		for (int k = 0; k < Lx; ++k) {
			RGB pixel_vec {*(data + pos), *(data + pos + 1), *(data + pos + 2)};
			pos += 3;
			lx_vec.emplace_back(pixel_vec);
		}
		ly_vec.emplace_back(lx_vec);
	}

	return ly_vec;
}



int main() {
	string raw_picture = "01358.npy";
	vector<vector<RGB>> figure_array = get_2d_space_rgb(raw_picture);
	cout << "the matrix has " << figure_array.size() << " raws" << endl;

	/*you could realize your algorithm down here*/
	vector<vector<double>> adjacency_matrix = spectral_clustering(vec2mat(figure_array), 10, 1);


	return 0;
}