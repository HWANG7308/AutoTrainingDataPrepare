import numpy as np
import cv2


def get_selection(
    options_list,
    text,
    multi_selection=False,
    select_from_range=False,
    with_return=True,
    with_exit=False,
):
    """
    Enables navigation in a list of options.

    Args:
        options_list (list): list of options
        text (string): text shown in TUI to guide selection

    Returns:
        string: some selection of the given list

    From https://github.com/KochPJ/AutoPoseEstimation
    """
    selection_string = ""
    if select_from_range:
        selection_string += "{}-{}  : select from range\n".format(
            options_list[0], options_list[-1]
        )
    else:
        for i, a in enumerate(options_list):
            selection_string += "{}   : {}\n".format(i + 1, a)

    if with_exit:
        selection_string += "exit   : exit program\n"
        options_list.append("exit")

    while True:
        # List options and get user input
        if with_return:
            selection = input(text + ":\n0   : return\n" + selection_string)
        else:
            selection = input(text + ":\n" + selection_string)

        if not multi_selection:
            single = True
        else:
            try:
                selection = list(selection)
                if len(selection) == 1:
                    single = True
                    selection = selection[0]
                else:
                    single = False
                    selections = []
                    n = ""
                    for s in selection:
                        if s in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]:
                            n += s
                        elif s == ",":
                            selections.append(int(n))
                            n = ""
                    if len(n) > 0:
                        selections.append(int(n))

                    if 0 in selections:
                        return None
                    uniques, counts = np.unique(selections, return_counts=True)
                    if len(uniques) != np.sum(counts):
                        print(
                            "Found the value for {} multiple times. Try again.".format(
                                [x for x in uniques[counts > 1]]
                            )
                        )
                        continue
                    out_of_scope = [
                        s for s in selections if s < 0 or s > len(options_list)
                    ]
                    if out_of_scope:
                        print("Found value out of scope for {}".format(out_of_scope))
                        continue
                    else:
                        return [options_list[s - 1] for s in selections]

            except ValueError:
                print("That is not a valid option. Try again.")
                continue

        if single:
            if with_exit:
                if selection == "exit":
                    return selection
            try:
                # Convert user input to index of options list
                selection = int(selection)
            except ValueError:
                # This is raised when there is input that cannot be converted to an integer
                print("That is not a valid option. Try again.")
                continue

            # If user input is 0, return None
            if selection == 0:
                return None

            # If user input is valid, return chosen option
            elif 0 < selection <= len(options_list):
                name = options_list[selection - 1]
                return name

            # If user input is not valid, let the user try again
            else:
                print("That is not a valid option. Try again.")


def draw_ArUco(marker_id=42, marker_size=200):

    # Define the dictionary and marker ID
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

    # Generate the marker
    marker_image = np.zeros((marker_size, marker_size), dtype=np.uint8)
    marker_image = cv2.aruco.generateImageMarker(
        aruco_dict, marker_id, marker_size, marker_image, 1
    )

    return marker_image


def draw_ChArUco():
    # Define ChArUco board parameters
    squares_x = 5  # Number of squares in X direction
    squares_y = 7  # Number of squares in Y direction
    square_length = 0.04  # Square side length (in meters)
    marker_length = 0.02  # Marker side length (in meters)

    # Create ChArUco board
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    charuco_board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y), square_length, marker_length, aruco_dict
    )

    # Generate ChArUco board image
    board_image = charuco_board.generateImage((600, 800))

    return board_image
