from cuda.bindings.nvml import EventData
from transformers import AutoTokenizer
import json

from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from llm_course.backends import append_event

@dataclass
class EventData:
    tokenizer: str
    data: list[TokenizerResult]

@dataclass
class TokenizerResult:
    text: str
    ids: list
    pieces: list
    count: int

def append_tokenizer_log(path: str, eventData: list[EventData]) -> None:
    result = list()
    for val in eventData:
        result.append(asdict(val))
    with open(path, "w", encoding="utf-8") as file:
        file.write(json.dumps(result, ensure_ascii=False) + "\n")


MODEL_NAME_V1 = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer_v1 = AutoTokenizer.from_pretrained(MODEL_NAME_V1)
MODEL_NAME_V2 = "bert-base-cased"
tokenizer_v2 = AutoTokenizer.from_pretrained(MODEL_NAME_V2)

samples = [
    "Sternik",
    "val filterWarningFlag: StateFlow<Boolean> = combine(\n" +
    "    weekPlanData,planActivitiesList,loadingStatus\n" +
    "    ){ weekData, activities, loadingStatus ->\n" +
    "        if(weekData == null) return@combine false\n" +
    "        val weekDataIsEmpty = weekData.values.none{it.intervals?.isNotEmpty() == true}\n" +
    "        weekDataIsEmpty && activities.isNotEmpty() && loadingStatus == LoadingStatus.DONE\n" +
    "    }.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000),false)",
    "1125",
    "🫶",
    "Pchnąć w tę łódź jeża lub ośm skrzyń fig."
]

results = list()
for name, tokenizer in [(MODEL_NAME_V1, tokenizer_v1), (MODEL_NAME_V2, tokenizer_v2)]:
    tokenizerResults = list()
    for text in samples:
        ids = tokenizer.encode(text, add_special_tokens=False)
        pieces = [tokenizer.decode([token_id]) for token_id in ids]
        print(f"\nTEXT:   {text!r}")
        print(f"IDS:    {ids}")
        print(f"PIECES: {pieces}")
        print(f"COUNT:  {len(ids)}")
        tokenizerResults.append(TokenizerResult(text, ids, pieces, len(ids)))
        #print(f"RECONSTRUCTED: {tokenizer.decode(ids)}")

    event = EventData(
        tokenizer = name,
        data = tokenizerResults
    )
    results.append(event)
append_tokenizer_log("lab01a.json",results)