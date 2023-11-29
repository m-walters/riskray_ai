import sys, time, os
import argparse
import cv2

from .. dicom.dataset_tools import get_dicom, get_s3_connection, get_dicom_image, set_dicom_image


# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        cropping = False
        # draw a rectangle around the region of interest
        cv2.rectangle(image, refPt[0], (x, y), (0, 255, 0), 2)
        cv2.imshow("image", image)


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--test", dest='test', action='store_true',
        help="Whether this is a dry run for testing. If False, save these to S3 bucket"
    )
    ap.add_argument(
        "-o", "--out-name", required=False, default=None,
        help="Give the new dicom files an output name. This will then have the index appended to it"
    )
    ap.add_argument(
        "-i", "--start-index", required=False, default=0,
        help="Starting counter index for file naming"
    )
    ap.add_argument(
        "-f", "--input", required=True,
        help="Path to a file that is a list of DICOM file names to process"
    )
    ap.set_defaults(test=False)
    args = vars(ap.parse_args())
    print(f"ARGS -- {args}")

    files = []
    with open(args['input'], 'r') as todo:
        files = [f.rstrip("\n") for f in todo.readlines()]

    new_dicom_root = 'dicom/croppedDICOMs_2'
    print(f"PROCESSING {len(files)} FILES:\n{files[0]}\n...\n{files[-1]}")
    client, bucket = get_s3_connection()

    idx = int(args['start_index'])
    quitting = False  # Gracefully handle the quit
    for path in files:
        skip = False
        dicom = get_dicom(client, path)
        image = get_dicom_image(dicom)
        original_type = image.dtype
        image = image.astype("float32")
        # We'll make this float for now and then reconvert to int
        print(f"Preparing image ({image.shape} px) from {path} ")
        img_max = image.max()
        image /= img_max
        clone = image.copy()

        cv2.namedWindow("image")
        cv2.setMouseCallback("image", click_and_crop)
        # keep looping until the 'q' key is pressed
        while True:
            # display the image and wait for a keypress
            cv2.imshow("image", image)
            key = cv2.waitKey(1) & 0xFF
            # if the 'r' key is pressed, reset the cropping region
            if key == ord("r"):
                image = clone.copy()
            # if the 'c' key is pressed, break from the loop
            elif key == ord("c"):
                break
            elif key == ord("q"):
                quitting = True
                break
            elif key == ord("n"):
                skip = True
                break

        if quitting:
            # Shutdown windows
            print("Exiting")
            cv2.destroyAllWindows()
            sys.exit(0)

        if skip:
            # Increase the counter to mark this
            print(f"Skipping {path}. Still increasing counter from {idx} to {idx+1}")
            idx += 1
            cv2.destroyAllWindows()
            refPt = []
            cropping = False
            continue

        # if there are two reference points, then crop the region of interest
        # from teh image and display it
        cropped = None
        if len(refPt) == 2:
            print(f'REFPT -- {refPt}')
            # Organize the ref-pts properly
            # The (x,y) pairs already start top-left like numpy
            left, right = refPt[0][0], refPt[1][0]
            if left > right:
                x = left
                left = right
                right = x

            # Here, top should actually be the lower number
            top, bottom = refPt[0][1], refPt[1][1]
            if top > bottom:
                y = top
                top = bottom
                bottom = y

            cropped = clone[top:bottom, left:right]
            cv2.imshow("ROI", cropped)
            cv2.waitKey(0)

        print(f"Done crop. New image shape: {cropped.shape}")
        if not args['test']:
            # Upload to S3
            # Save new dicom
            # Don't forget to re-multiply by img_max
            cropped *= img_max
            # Re-cast back to uint16
            cropped = cropped.astype(original_type)
            dicom = set_dicom_image(dicom, cropped)
            local_savename = f"{os.getcwd()}/tmp_dicom"
            dicom.save_as(local_savename, write_like_original=True)
            time.sleep(2)

            if args['out_name']:
                outpath = f"{args['out_name']}{idx}"
            else:
                outpath = path[6:]
            outpath = f"{new_dicom_root}/{outpath}"
            print(f"SAVING CROPPED DICOM TO S3 TO PATH:\n\t{outpath}")
            client.upload_file(local_savename, 'riskraydata', outpath)
            time.sleep(1.5)
            os.remove(local_savename)

        # close all open windows
        cv2.destroyAllWindows()
        refPt = []
        cropping = False
        idx += 1

    print(f"DONE")
    sys.exit(0)
