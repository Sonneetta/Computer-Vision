#include <opencv2/opencv.hpp>
#include "func.hpp"
#include "task.hpp"

using namespace cv;
using namespace std;

int main()

{
	int subtask;
	int taskNumber;

	do {
		cout << "Select a task to execute:" << endl;
		cout << "3. Task 3: Find and output the centroid, major axis, and eccentricity" << endl;
		cout << "5. Task 5: Estimate the curvature of the contour at different points" << endl;
		cout << "7. Task 7: Find and visualize lines in an image using a standard Hough transform " << endl;
		cout << "0. Exit" << endl;

		cin >> taskNumber;

		switch (taskNumber) {

		case 3: {

			do {
				cout << "1. Display centroid, major axis, and eccentricity" << endl;
				cout << "0. Go back to main menu" << endl; 

				cin >> subtask; 

				if (subtask != 0) { 
					task3(subtask); 
				}

			} while (subtask != 0); 
			break; 

		}
			
			break;

		case 5: {
	
			do {
				cout << "1. Display the maximum, minimum the average value of curvature(its absolute value)." << endl;
				cout << "0. Go back to main menu" << endl; 

				cin >> subtask; 

				if (subtask != 0) { 
					task5(subtask); 
				}

			} while (subtask != 0);
			break;
		}
			break;
		case 7: {

			do {
				cout << "1. Display lines in an image using a standard Hough transform" << endl;
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