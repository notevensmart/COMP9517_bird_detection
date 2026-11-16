"""Manual dataset integrity safeguards for classical bird detection.

This is NOT replacement annotation data. A flag means a human observed likely
target birds despite the supplied empty annotation; the file is unsafe as a
source of negative training patches until reviewed. Official VAL labels remain
unchanged for primary benchmark comparability. No TEST image is inspected.
"""
# Image IDs are visual QA flags, not inferred model labels.
SUSPECT_TRAIN_EMPTY_LABELS = {
    "beach_bird_0149.jpg": "Visible flying flock; target scope requires review",
    "beach_bird_0418.jpg": "Two visible shorebirds on wet sand",
}
SUSPECT_VAL_EMPTY_LABELS = {
    "beach_bird_1435.jpg": "Numerous visible shorebirds on wet sand",
}

def unsafe_training_negative(image_name, ground_truth_count):
    return ground_truth_count == 0 and image_name in SUSPECT_TRAIN_EMPTY_LABELS

def tests():
    assert unsafe_training_negative("beach_bird_0418.jpg", 0)
    assert unsafe_training_negative("beach_bird_0149.jpg", 0)
    assert not unsafe_training_negative("beach_bird_0418.jpg", 1)
    assert not unsafe_training_negative("beach_bird_1379.jpg", 0)
    print("PASS human QA quarantine test; val/test never filtered",flush=True)

if __name__=="__main__":tests()
