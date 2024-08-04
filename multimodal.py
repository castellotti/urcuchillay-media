#!/usr/bin/env python3
# Copyright (c) 2024 Steve Castellotti
# This file is part of Urcuchillay and is released under the MIT License.
# See LICENSE file in the project root for full license information.

import argparse
import logging
import os
import sys
import warnings


try:
    import config
    import utils
    from radient import make_operator
    from radient import Workflow

    from transformers import ChameleonProcessor, ChameleonForConditionalGeneration
    import torch
    from PIL import Image

    # The following modules need to be explicitly imported
    import numpy as np
    import pymilvus
    import accelerate
except ModuleNotFoundError as e:
    print('\nError importing Python module(s)')
    print('If installed using setup.sh it may be necessary to run:\n')
    print('pyenv activate urcuchillay-env\n')
    sys.exit(1)

INTERVAL = 2.0

SOURCE = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
PROMPT = "Is there a backflip?"


class Multimodal:
    def __init__(self, args):

        level = logging.DEBUG if args.debug else logging.INFO
        logging.basicConfig(stream=sys.stdout, level=level)
        logging.getLogger().name = __name__ if __name__ != '__main__' else 'multimodal'

        self.database_uri = config.Config.MILVUS_URI
        self.collection_name = config.Config.COLLECTION_NAME

        self.source = args.source
        self.prompt = args.prompt
        self.interval = args.interval

        # Set Parallel Iterator
        os.environ['TOKENIZERS_PARALLELISM'] = 'true' if config.Config.TOKENIZERS_PARALLELISM else 'false'

    def index(self):

        if os.path.exists(self.source):
            print("Processing: %s" % self.source)
            read = make_operator(optype="source", method="local", task_params={"path": self.source})
        else:
            print("Downloading from YouTube: %s" % self.source)
            read = make_operator(optype="source", method="youtube", task_params={"url": self.source})

        demux = make_operator(optype="transform", method="video-demux", task_params={"interval": self.interval})

        vectorize = make_operator(optype="vectorizer", method="imagebind", modality="multimodal", task_params={})

        store = make_operator(optype="sink", method="milvus",
                              task_params={
                                  "operation": "insert",
                                  "milvus_url": self.database_uri,
                                  "collection_name": self.collection_name})

        insert_wf = (Workflow()
                     .add(read, name="read")
                     .add(demux, name="demux")
                     .add(vectorize, name="vectorize")
                     .add(store, name="store")
                     )
        insert_wf()

    def search(self):

        vectorize = make_operator("vectorizer", "imagebind", modality="text")
        search = make_operator("sink", "milvus",
                               task_params={
                                   "operation": "search",
                                   "milvus_url": self.database_uri,
                                   "collection_name": self.collection_name,
                                   "output_fields": None})

        search_wf = (Workflow()
                     .add(vectorize, name="vectorize")
                     .add(search, name="search")
                     )

        search_vars = {
            "limit": 1,
            "output_fields": ["data", "modality"],
        }
        results = search_wf(
            extra_vars={"search": search_vars},
            data=self.prompt
        )

        filename = results[0][0][0]["entity"]["data"]

        processor = ChameleonProcessor.from_pretrained("nopperl/chameleon-7b-hf")

        if utils.is_mac():
            model = ChameleonForConditionalGeneration.from_pretrained(
                "nopperl/chameleon-7b-hf", device_map="cpu")
        else:
            model = ChameleonForConditionalGeneration.from_pretrained(
                "nopperl/chameleon-7b-hf", device_map="cpu", torch_dtype=torch.bfloat16)

        image = Image.open(filename)
        prompt = f"{self.prompt}<image>"

        if utils.is_mac():
            inputs = processor(prompt, image, return_tensors="pt").to(model.device)
        else:
            inputs = processor(prompt, image, return_tensors="pt").to(model.device, torch.bfloat16)

        out = model.generate(**inputs, max_new_tokens=50, do_sample=True)
        generated_text = processor.batch_decode(out, skip_special_tokens=False)[0]
        return image, generated_text

    @staticmethod
    def display(image, generated_text):
        # Print the generated text
        output = generated_text.replace('<image>', '')
        print(output)

        # Display the image using Pillow
        image.show()

    def reset(self):
        pymilvus.Collection(name=self.collection_name).drop()
        print(f"Collection {self.collection_name} cleared.")

    def run(self):
        self.index()
        image, generated_text = self.search()
        self.display(image, generated_text)


def setup_warning_filters():
    warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.transforms._functional_video")
    warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.transforms._transforms_video")


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process command parameters')
    parser = utils.parse_arguments_common(parser)

    parser.add_argument('--source', type=str, default=SOURCE,
                        help='The name of the source file or YouTube URL to index (default: %(default)s)')
    parser.add_argument('--prompt', type=str, default=PROMPT,
                        help='The prompt to search (default: %(default)s)')
    parser.add_argument('--interval', type=float, default=INTERVAL,
                        help='The time interval to index the source (default: %(default)s)')

    args = parser.parse_args()
    return args


def main():
    setup_warning_filters()
    args = parse_arguments()
    multimodal = Multimodal(args=args)
    multimodal.run()


if __name__ == "__main__":
    main()
