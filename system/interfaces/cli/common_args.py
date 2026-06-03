# -*- coding: utf-8 -*-

def _register_common_args(parser):
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--modes", nargs="+", default=["text", "image", "audio", "video"] )
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--audio", type=str, default=None)
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--use-local-models", action="store_true")
    parser.add_argument("--local-text-model", type=str, default="")
    parser.add_argument("--local-image-model", type=str, default="")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--pure-chat", action="store_true")

