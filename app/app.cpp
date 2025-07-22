#include <opencv2/opencv.hpp>
#include "func.hpp"
#include "task.hpp"

using namespace cv;
using namespace std;


int main() {

	int subtask;
	int taskNumber;

	do {
		cout << "Select a task to execute:" << endl;
		cout << "1. Task 1: Histogram Equalization" << endl;
		cout << "3. Task 3: Linear local operator (given filter kernel)" << endl;
		cout << "5. Task 5: Unsharp macking" << endl;
		cout << "7. Task 7: Harris and FAST angle detectors" << endl;
		cout << "0. Exit" << endl;

		cin >> taskNumber;

		switch (taskNumber) {
		case 1: {

			do {
				cout << "Select what you want to display:" << endl;
				cout << "1. Display histograms" << endl;
				cout << "2. Display images" << endl;
				cout << "0. Go back to main menu" << endl;
				cin >> subtask;

				if (subtask != 0) {
					task1(subtask);
				}

			} while (subtask != 0);
			break;
		}

		case 3:
			do {
				cout << "Select what you want to display:" << endl;
				cout << "1. Display images blur" << endl;
				cout << "2. Display images Sobel_x" << endl;
				cout << "0. Go back to main menu" << endl;
				cin >> subtask;

				if (subtask != 0) {

					task3(subtask);
				}

			} while (subtask != 0);
			break;
		case 4:

			break;
		case 5: {

			do {
				cout << "Select what you want to display:" << endl;
				cout << "1. Display images box filter" << endl;
				cout << "2. Display images median filter" << endl;
				cout << "3. Display images gaussian filter" << endl;
				cout << "4. Display images sigma filter" << endl;
				cout << "5. Show all sharpened images" << endl;
				cout << "0. Go back to main menu" << endl;
				cin >> subtask;

				if (subtask != 0) {

					task5(subtask);
				}

			} while (subtask != 0);

		}
			  break;
		case 7: {

			do {
				cout << "Select what you want to display:" << endl;
				cout << "1. Display images HARRIS" << endl;
				cout << "2. Display images FAST" << endl;
				cout << "0. Go back to main menu" << endl;
				cin >> subtask;

				if (subtask != 0) {

					task7(subtask);
				}

			} while (subtask != 0);

		}
			  break;
		case 0:
			cout << "Exiting the program." << endl;
			break;
		default:
			cout << "Invalid task number!" << endl;
			break;
		}

	} while (taskNumber != 0);

	return 0;
}

