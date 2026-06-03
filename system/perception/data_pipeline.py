from typing import Dict, Any


class DataPipeline:
    """Minimal data pipeline to prepare payloads for GNN updates and LLM tuning.

    This lightweight implementation converts `result` and contextual info into
    small dicts that downstream components (buffers, trainers) can consume.
    """

    def __init__(self, system=None):
        self.system = system

    def prepare_gnn_data(self, result: Dict[str, Any], best_action: Any, feedback_score: float) -> Dict[str, Any]:
        if not result:
            return None
        return {
            'summary': {
                'nodes_created': result.get('nodes_created', 0),
                'gnn_loss': result.get('gnn_loss', 0.0),
                'feedback_score': feedback_score,
            },
            'action': best_action,
            'timestamp': __import__('time').time(),
        }

    def prepare_tuning_data(self, result: Dict[str, Any], best_action: Any, feedback_score: float) -> Dict[str, Any]:
        if not result:
            return None
        return {
            'examples': {
                'input_example': result.get('example_input', None),
                'output_example': result.get('example_output', None),
            },
            'meta': {
                'feedback_score': feedback_score,
                'action': best_action,
                'timestamp': __import__('time').time(),
            }
        }
