"""Dependencies and runtime settings for technique transfer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from research_assistant import config
from research_assistant.transfer.cards import CardIndex
from research_assistant.transfer.models import PriorArt


@dataclass
class Deps:
    chat: Callable[[list[dict], float], str]
    critic: Callable[[list[dict]], str]
    judge: Callable[[str, str], dict]
    embed_query: Callable[[str], list[float]]
    embed_documents: Callable[[list[str]], list[list[float]]]
    shortlist: Callable[[str, int], list[dict]]
    gate: Callable[[str, list[dict]], list[dict]]
    search: Callable[[str, int, set | None, set | None], list[dict]]
    prior_art: Callable[[str, list[str], str, int | None], PriorArt]
    texts: list[str]
    metadatas: list[dict]
    cards: CardIndex
    cancelled: Callable[[], bool] = lambda: False
    progress: Callable[[int, int, str], None] = lambda d, t, label: None


@dataclass
class TransferSettings:
    bottlenecks: int = config.TRANSFER_BOTTLENECKS
    cards_per_query: int = config.TRANSFER_CARDS_PER_QUERY
    techniques_per_bottleneck: int = config.TRANSFER_TECHNIQUES_PER_BOTTLENECK
    near_passages: int = config.TRANSFER_NEAR_PASSAGES
    source_passages: int = config.TRANSFER_SOURCE_PASSAGES
    passage_max_chars: int = config.TRANSFER_PASSAGE_MAX_CHARS
    min_near_papers: int = config.TRANSFER_MIN_NEAR_PAPERS
    temperature: float = config.TRANSFER_TEMPERATURE
    bottleneck_temperature: float = config.TRANSFER_BOTTLENECK_TEMPERATURE
    exclude_documents: frozenset = frozenset()
    prior_art_before_year: int | None = None
