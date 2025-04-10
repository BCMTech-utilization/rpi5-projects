import os
from transformers import AutoProcessor, AutoConfig
import onnxruntime as ort
import time
import numpy as np
import cv2
from tokenizers import Tokenizer as TokenizerFast
from hailo_platform import VDevice, FormatType, HailoSchedulingAlgorithm
import clip
import torch
from my_camera import Camera
from timer import Timer
import argparse


CAPTION_EMBEDDING = "resources/embeddings/caption_embedding.npy"
WORD_EMBEDDING = "resources/embeddings/word_embedding.npy"
ENCODER_PATH = "resources/models/florence2_transformer_encoder.hef"
DECODER_PATH = "resources/models/florence2_transformer_decoder.hef"
VISION_ENCODER_PATH = "resources/models/vision_encoder.onnx"
DECODER_INPUT_SHAPE = (1, 1, 32, 768)
ENCODER_OUTPUT_KEY = "resources/transformer_decoder/input_layer1"
DECODER_INPUT_KEY = "resources/transformer_decoder/input_layer2"
TOKENIZER_PATH = "resources/tokenizer/tokenizer.json"
START_TOKEN = 2
TIMEOUT_MS = 1000
COSINE_SIMILARITY_THRESHOLD = 0.7
GLOBAL_TIMER= Timer()

def argparser():
    parser = argparse.ArgumentParser(description="Configurations for Flourence.")
    parser.add_argument("--video", type=str, default="/dev/video0")
    parser.add_argument('--no-speaker', action="store_true", help='Use this flag in case you did not connected a speaker')

    return parser.parse_args()

def match_texts(model, text1, text2):
    # Load the CLIP model and preprocess function
    start= time.time()
    # Tokenize the texts and encode them into embeddings
    texts = [text1, text2]
    text_inputs = clip.tokenize(texts)

    # Get the text features (embeddings) from CLIP
    text_features = model.encode_text(text_inputs)

    # Normalize the features to unit vectors (important for similarity comparisons)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Compute cosine similarity between the two text embeddings
    similarity = torch.cosine_similarity(text_features[0], text_features[1], dim=0)
    #print (f"similarity between '{text1}' and '{text2}' is: {similarity.item()}")
    print(f"last, match_text elapsed, {round(time.time()- start, 4)}")
    return similarity.item()



def create_processor():
    size={'height':384, 'width':384}
    return AutoProcessor.from_pretrained('microsoft/florence-2-base', trust_remote_code=True, size=size, crop_size=size)

def infer_davit(inputs, davit_session):
    timer= Timer()
    image_features = davit_session.run(None, {'pixel_values':inputs.pixel_values.numpy()})[0]
    timer.record("infer davit")
    return image_features

def infer_encoder(encoder, image_text_embeddings):
    timer= Timer()
    encoder_hidden_state = np.empty((1,153,768), dtype=np.float32)
    bindings = encoder.create_bindings()
    bindings.input().set_buffer(image_text_embeddings)
    bindings.output().set_buffer(encoder_hidden_state)
    job = encoder.run_async([bindings], lambda completion_info: None)
    job.wait(TIMEOUT_MS)
    timer.record("infer encoder")
    return encoder_hidden_state

def infer_decoder(decoder, encoder_output, input_embeds):
    timer= Timer()
    decoder_output = np.empty((32,51289), dtype=np.float32)
    bindings = decoder.create_bindings()
    bindings.input('florence2_transformer_decoder/input_layer1').set_buffer(encoder_output)
    bindings.input('florence2_transformer_decoder/input_layer2').set_buffer(input_embeds)
    bindings.output().set_buffer(decoder_output)
    job = decoder.run_async([bindings], lambda completion_info: None)
    job.wait(TIMEOUT_MS)
    timer.record("infer_decoder")
    return decoder_output

def infer_florence2(image, processor, davit_session, encoder, decoder, tokenizer):
    timer= Timer()
    inputs = processor(text='<CAPTION>', images=image, return_tensors='pt')
    timer.record("first, processor elapsed", reset=True)
    image_features = infer_davit(inputs, davit_session)
    image_text_embeddings = np.concatenate([np.expand_dims(image_features, axis=0), np.load(CAPTION_EMBEDDING)], axis=2)
    encoder_hidden_state = infer_encoder(encoder, image_text_embeddings)
    
    word_embedding = np.load(WORD_EMBEDDING)
    decoder_input = np.insert(np.zeros(DECODER_INPUT_SHAPE).astype(np.float32), 0, word_embedding[START_TOKEN], axis=2)[:, :, :-1, :]
    dataset = {
        ENCODER_OUTPUT_KEY : encoder_hidden_state,
        DECODER_INPUT_KEY : decoder_input
    }
    next_token_id = -1
    token_index = 0
    generated_ids = [START_TOKEN]
    timer.reset()
    while next_token_id != START_TOKEN and token_index < 32:
        decoder_output = infer_decoder(decoder, dataset[ENCODER_OUTPUT_KEY], dataset[DECODER_INPUT_KEY])
        res = decoder_output.squeeze()[token_index]
        next_token_id = np.argmax(res)
        token_index += 1
        generated_ids.append(next_token_id)
        decoder_input = np.insert(decoder_input, token_index, word_embedding[next_token_id], axis=2)[:, :, :-1, :]
        dataset[DECODER_INPUT_KEY] = decoder_input
    res = tokenizer.decode(np.array(generated_ids), skip_special_tokens=True)
    timer.record("fourth, decode elapsed")
    return res

def caption_loop(camera, processor, davit_session, encoder, decoder, tokenizer, clip_model, no_speaker):
    last_caption = None
    timer= Timer()
    while True:
        camera.reset()
        is_capture_success, image= camera.get_picture()
        if not is_capture_success :
            print("is not capture success")
            continue

        camera.show_picture(image)
        caption = infer_florence2(image, processor, davit_session, encoder, decoder, tokenizer)
        if last_caption is None or match_texts(clip_model, last_caption, caption) < COSINE_SIMILARITY_THRESHOLD:
            print(f"NEW EVENT ALERT!!!!! - {caption}")
            if not no_speaker:
                os.system(f'espeak "{caption}" -s 130')
        last_caption = caption
        timer.record("one loop", reset=True)

def main():
    print("Initializing...")
    timer= Timer()
    args = argparser()
    processor = create_processor()
    davit_session = ort.InferenceSession(VISION_ENCODER_PATH)
    tokenizer = TokenizerFast.from_file(TOKENIZER_PATH)
    clip_model, _ = clip.load("ViT-B/32", "cpu")
    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN    
    timer.record("let's make camera")
    camera= Camera(args.video)
    with VDevice(params) as vd:
        encoder_infer_model = vd.create_infer_model(ENCODER_PATH)
        encoder_infer_model.input().set_format_type(FormatType.FLOAT32)
        encoder_infer_model.output().set_format_type(FormatType.FLOAT32)
        with encoder_infer_model.configure() as encoder:
            decoder_infer_model = vd.create_infer_model(DECODER_PATH)
            decoder_infer_model.input('florence2_transformer_decoder/input_layer1').set_format_type(FormatType.FLOAT32)
            decoder_infer_model.input('florence2_transformer_decoder/input_layer2').set_format_type(FormatType.FLOAT32)
            decoder_infer_model.output().set_format_type(FormatType.FLOAT32)
            with decoder_infer_model.configure() as decoder:
                print("Initialized succesfully")
                timer.record("program start")
                caption_loop(camera, processor, davit_session, encoder, decoder, tokenizer, clip_model, args.no_speaker)
                    

if __name__=="__main__":
    main()
