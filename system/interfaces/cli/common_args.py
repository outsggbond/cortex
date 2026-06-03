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
    parser.add_argument(
        "--api-server",
        action="store_true",
        help="Start HTTP API server (default: http://127.0.0.1:8000). Use with --api-port and --api-host.",
    )
    parser.add_argument(
        "--api-host",
        type=str,
        default="127.0.0.1",
        help="HTTP API server bind address (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=8000,
        help="HTTP API server port (default: 8000).",
    )

