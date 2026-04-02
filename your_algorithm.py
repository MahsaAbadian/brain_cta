"""
The following is a simple example algorithm the TopBrain multiclass segmentation.

It can be run locally (good for setting up and debugging your algorithm)
or in a docker container (required for submission to grand-challenge.org).

If you use this template you can simply update the `your_segmentation_algorithm` function with your own algorithm.
The input is the np.array in (x,y,z) extracted from the MR or the CT image (depending on your TRACK), the output is your segmentation prediction array.

To run your algorithm, execute the `inference.py` script (inference.py is the entry point for the docker container).
NOTE: You don't need to change anything in the inference.py script!

The relevant paths are as follows:
    input_path: contains the input images inside the folder /images/head-mr-angio or /images/head-ct-angio (depending on your TRACK)
        Docker: /input
        Local: ./test/input
    output_path: output predictions are stored inside the folder /images/head-mr-angio-segmentation or /images/head-ct-angio-segmentation (depending on your TRACK)
        Docker: /output
        Local: ./test/output
    resource_path: any additional resources needed for the algorithm during inference can be stored here (optional)
        Docker: resources
        Local: ./resources

Before submitting to grand-challenge.org, you must ensure that your algorithm runs in the docker container. To do this, run
  bash ./test_run.sh
This will start the inference and reads from ./test/input and outputs to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:
  bash ./save.sh
"""

import numpy as np

#######################################################################################
# TODO-1:
# Choose your TRACK. Track is either 'MR' or 'CT'.
TRACK = "CT"  # or 'CT'
# END OF TODO-1
#######################################################################################


#######################################################################################
# TODO-2:
# if you use pytorch, we have a util to show torch and cuda status
do_you_use_pytorch_cuda = False  # True or False
# END OF TODO-2
#######################################################################################


def your_segmentation_algorithm(*, input_array: np.array) -> np.array:
    """
    This is an example of a prediction algorithm.
    It is a dummy algorithm that returns an array of the correct shape filled with ones.
    args:
        input_array: np.array (x,y,z) from the input image
    returns:
        np.array - prediction in the same shape as the input_array
    """

    #######################################################################################
    # TODO-3: place your own prediction algorithm here.
    # You are free to remove everything! Just return to us an npy in (x,y,z).
    # Depending on the value of TRACK, the input_array will either be the MR or the CT image for you to segment.

    # NOTE: the prediction array must have the same shape as the input image!

    # NOTE: If you extract the array from SimpleITK, note that
    #              SimpleITK npy array axis order is (z,y,x).
    #              We have already transposed SimpleITK.GetArrayFromImage to (x,y,z) for you!

    # NOTE: You can also put files in the `resources` folder (e.g. model weights),
    # Remember to copy them in the Dockerfile!
    # COPY --chown=user:user resources /opt/app/resources

    # load and initialize your model here
    # model = ...
    # device = ...

    # You can also place and load additional files in the resources folder
    # with open(resources / "some_resource.txt", "r") as f:
    #     print(f.read())

    # for example, we can return all 1s as prediction
    pred_array = np.ones(input_array.shape)
    # replace the above line with your actual algorithm

    # END OF TODO-3
    #######################################################################################

    return pred_array
