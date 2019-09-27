# -*- coding: utf-8 -*-
#
# Copyright © 2019 Attila Ulbert
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation 
# files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, 
# modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software 
# is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE 
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR 
# IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import codecs
import argparse

from keras.models import load_model


def run(weights_file, model_file, output_file):
    print("Loading model file.")
    model = load_model(model_file)
    print("Loading weights file.")
    model.load_weights(weights_file)

    print("Saving merged file.")
    model.save(output_file)

    print('\nFinished merging!')


def main():
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--weights_file",
        type=str,
        help="Weight file.")
    parser.add_argument(
        "--model_file",
        type=str,
        help="Model file.")
    parser.add_argument(
        "--output_file",
        type=str,
        help="Model and weight file.")

    flags, unparsed = parser.parse_known_args()

    print("weights_file: '" + flags.weights_file + "'")
    print("model_file: '" + flags.model_file + "'")
    print("output_file: '" + flags.output_file + "'")

    run(flags.weights_file, flags.model_file, flags.output_file)


if __name__ == '__main__':
    if sys.stdout.encoding != 'cp850':
        sys.stdout = codecs.getwriter('cp850')(sys.stdout.buffer, 'strict')
    if sys.stderr.encoding != 'cp850':
        sys.stderr = codecs.getwriter('cp850')(sys.stderr.buffer, 'strict')

    main()
