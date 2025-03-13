## Description

1. Create an account on https://app.platerecognizer.com/accounts/signup/
2. Call the API from https://docs.platerecognizer.com/#license-plate-recognition to find the license plate in the `images` directory.
3. For each license plate, get the bounding box of the licence plate.
4. Then calculate the background color of license plate from the pixels.
5. Use that color to draw a bounding box of the license plate. Use a bounding box thickness of 4 pixels.
6. Save the image to `output/example.jpg`.

Notes:
- Your solution will be judged based on readability, correctness and documentation.
- This task should not take more than 1-2 hours.

## Submission

- Create a zip file exercise.zip
- After it's unzipped, it should contain the project source code.
- The program MUST run inside Docker. We will run your script using the following command:

```shell
unzip exercise.zip
cd exercise
docker build -t exercise .
docker run -v /path/to/exercise:/app exercise
ls output/*jpg # images created by the program
```
