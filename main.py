from predict import get_prediction
from input.parse_input import parse_data
from report.generate_report import generate_html

import os

def main(input_dir: str):
    meeting_info = parse_data(input_dir)
    audio_label = get_prediction(os.path.join(input_dir, 'audio.wav'), 'model.h5')
    # generate_html

if __name__ == '__main__':
    input_dir = ''
    main(input_dir)