# TopBrain 2025 Algorithm Submission Template 🔝🧠

This repo is the **algorithm submission template** for the [**TopBrain 2025 challenge**](https://topbrain2025.grand-challenge.org) on grand-challnge (GC).
With a few lines of code, you can modify this repo to generate the Docker container and take part in the TopBrain challenge!

## Important Notes

* **If you want more flexibility (i.e. customize this repo beyond `your_algorithm.py` file or not using this template repo at all)**, then simply ensure that your docker container can:
  1. read from the input interface that supplies **one `.mha` image**:

        > Head MR Angiography (Image) at **`/input/images/head-mr-angio/<uuid>.mha`**
        >
        > Head CT Angiography (Image) at **`/input/images/head-ct-angio/<uuid>.mha`**

  2. write to the output interface **one `.mha` segmentation mask**:

        > Head MR Angiography Segmentation to **`/output/images/head-mr-angio-segmentation/<uuid>.mha`**
        >
        > Head CT Angiography Segmentation to **`/output/images/head-ct-angio-segmentation/<uuid>.mha`**


* **If you are new to the GC submission system**, we recommend that you clone this repo locally and use it as a template, **which only requires a few simple steps to build a submission container!**
The steps are easily done, and you just follow the `TODO` in one file!

    1. Clone this repo locally
    2. Add your algorithm by editing the **`./your_algorithm.py`** file
    3. Submit your algorithm by
        - `bash ./save.sh` to create a `tar.gz` of your docker container and upload to GC
        - see GC documentation "Option 2 (Uploading the container image)": [how to deploy your container](https://grand-challenge.org/documentation/test-and-deploy-your-container/)


  For more information on **using this repo as a template, please refer to [Use This Repo as a Template](#use-this-repo-as-a-template)** below.


* **Submission portal links**. We have four submission portals: Two submission portals for the validation phase, and two for the final-test phase, one for each track:
    * Submit to [Validation CTA](https://topbrain2025.grand-challenge.org/evaluation/validation-cta-seg/submissions/create/)
    * Submit to [Validation MRA](https://topbrain2025.grand-challenge.org/evaluation/validation-mra-seg/submissions/create/)
    * Submit to [Final Test CTA](https://topbrain2025.grand-challenge.org/evaluation/finaltest-cta-seg/submissions/create/)
    * Submit to [Final Test MRA](https://topbrain2025.grand-challenge.org/evaluation/finaltest-mra-seg/submissions/create/)

    Please use the validation submission portals to make sure your docker containers work as intended.
    You can submit to validation phases with a daily limit until they close.
    But final test phases only allow for **one** submission.

---

## Use This Repo as a Template

### Edit `./your_algorithm.py`

You can adapt the `your_algorithm.py` file. We have marked the most relevant parts you need to change with **`TODO`**.

For each test case, the output of your algorithm must be a prediction array either for the CT or the MR image (depending on the track).

Simply specify `TRACK` on top of the `your_algorithm.py` file:

```python
# TODO: 
# Choose your TRACK. Track is either 'MR' or 'CT'.
TRACK = 'MR' # or 'CT'
```

Finally, in the `your_segmentation_algorithm()` function, implement your inference algorithm there, and whatever you do,
**just return us an `numpy array`** of the same shape as the input image. We will handle the rest of the file conversion and output saving etc from there onwards.

```python
def your_segmentation_algorithm(*, mr_input_array: np.array, ct_input_array: np.array) -> np.array:
    """
    args:
        mr_input_array: np.array - input image for MR track
        ct_input_array: np.array - input image for CT track
    returns:
        np.array - prediction
    """

    # TODO: place your own prediction algorithm here
    model = ...
    device = ...
    ...
    model.predict(ct_input_array)
    ...
    # END OF TODO

    # return prediction array
    return pred_array
```

#### Running inference

You can run inference locally by executing the script `inference.py`. The `inference.py` also serves as the entrypoint for the Docker container. 

**NOTE: You don't need to change anything in the `inference.py` script.**

### Testing and deploying Docker container

Update your `requirements.txt` for your required python libraries.

Make the necessary changes to the `Dockerfile`:

* Choose a suitable base image if necessary to build your container (e.g., Tensorflow or Pytorch or even Ubuntu, or a different version of Python base image)
* Make sure that all required source files (such as model weights and python scripts) are copied to the Docker container with `COPY` in your `Dockerfile`

```docker
COPY --chown=user:user <somefile> /opt/app/
```

#### **Test your container!**

**Highly recommended to test your container by `bash test_run.sh` locally**. This will mimic the GC docker running environment and input to your docker container any mha files you provide in the `./test/input` folder. It will check the output predictions against what you provide in `./test/expected_output/`:

```bash
# in ./test_run.sh
# TODO: Provide the expected output segmentation mask of your algorithm in ./test/expected_output/
# TODO: In the python code snippet below change the following if necessary:

TRACK="MR"  # or "CT"
EXPECTED_SEG_MASK="expected_output_dummy_mra.mha"
```

**Change the input test images in `test/input/images/head-ct-angio/` or `test/input/images/head-mr-angio/`, and the expected output in `test/expected_output` to validate your algorithm works in the form of a Docker container**.

**Note:** the GC environment will process the test images sequentially one image at a time. So, there should only be 1 CT image or 1 MR image in the corresponding input folders.

Currently, you find `test/input/images/head-ct-angio/example_input_dummy_cta.mha` and `test/input/images/head-mr-angio/example_input_dummy_mra.mha` in the input folder.

Please note that GC environment has a limit on
* Main memory of at most 31 GiB DRAM ⚠️
  * The `test_run.sh` simulates this limitation locally. So make sure you test your container with our `test_run.sh` first!
* Container size must be under 10GB ⚠️

### Export and Deploy

If you choose to upload your container to GC directly (instead of linking a private Github repo, see above TLDR), then run `save.sh` to package your docker container image. This will create a `.tar.gz` file ready for upload to one of our GC submission portals.

**NOTE: it is a good idea to rename the default generated file with a more informative name, since we have 2 tracks:**

```bash
# e.g.
mv topbrain_algo_docker_whattrack_seg_<timestamp>.tar.gz <some_info>_<track>_tropbain_<timestamp>.tar.gz
```

### Making a Challenge Submission

Please refer to the GC documentation on
* ["Submitting your Algorithm container"](https://grand-challenge.org/documentation/making-a-challenge-submission/#submitting-your-algorithm-container)
  * espeically the **["Submission tips"](https://grand-challenge.org/documentation/making-a-challenge-submission/#submission-tips)**.

_NOTE: It is **recommended to use the GC to "Try Out Algorithm" first**:_
> Once your container is active, please test it out before submitting it to the challenge, you can upload data using the "Try Out Algorithm" button. Please use a representative image and check that the outputs are what you expect. You can inspect the error messages on the Results page of your algorithm.

---

And that is it! 🤠
All the best for the submission process.
Please reach out to us by leaving an issue on this repo or on our challenge forum.
