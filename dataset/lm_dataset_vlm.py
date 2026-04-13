import json
import random
import re
from bisect import bisect_right
from collections import OrderedDict
from io import BytesIO
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None


IMAGE_TAG_RE = re.compile(r"<image>|<img>")
SYSTEM_PROMPTS = [
    "You are a helpful AI assistant.",
    "You are minimind, a lightweight intelligent assistant.",
    "You are a knowledgeable AI. Try your best to provide accurate information.",
    "You are careful, concise, and accurate.",
    "You are a multimodal assistant that helps with image-grounded tasks.",
]


def _maybe_json_loads(value):
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    if text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _ensure_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _load_records(data_path: Path):
    suffix = data_path.suffix.lower()
    if suffix == ".parquet":
        if pq is None:
            raise RuntimeError(
                "Reading parquet datasets requires `pyarrow`. "
                "Install pyarrow or convert the dataset to json/jsonl."
            )
        return ParquetRecordStore(data_path)

    if load_dataset is not None:
        if suffix in {".json", ".jsonl"}:
            return load_dataset("json", data_files=str(data_path), split="train")

    if suffix == ".jsonl":
        with data_path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if suffix == ".json":
        with data_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]
    raise RuntimeError(
        f"Unsupported dataset format for {data_path}. "
        "Install `datasets` for parquet support or use json/jsonl."
    )


class ParquetRecordStore:
    def __init__(
        self,
        data_path: Path,
        columns=("conversations", "image_bytes", "image_names"),
        batch_size=1,
        cache_size=8,
    ):
        self.data_path = Path(data_path).expanduser().resolve()
        self.columns = list(columns)
        self.batch_size = batch_size
        self.cache_size = cache_size
        self._parquet_file = None
        self._cache = OrderedDict()

        parquet_file = pq.ParquetFile(str(self.data_path))
        self.row_group_sizes = [
            parquet_file.metadata.row_group(idx).num_rows
            for idx in range(parquet_file.num_row_groups)
        ]
        self.row_group_offsets = []
        offset = 0
        for size in self.row_group_sizes:
            self.row_group_offsets.append(offset)
            offset += size
        self.total_rows = offset

    def __len__(self):
        return self.total_rows

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_parquet_file"] = None
        state["_cache"] = OrderedDict()
        return state

    def _ensure_open(self):
        if self._parquet_file is None:
            self._parquet_file = pq.ParquetFile(str(self.data_path))

    def _locate_row(self, index):
        if index < 0:
            index += self.total_rows
        if index < 0 or index >= self.total_rows:
            raise IndexError(f"Index out of range: {index}")

        row_group_idx = bisect_right(self.row_group_offsets, index) - 1
        local_index = index - self.row_group_offsets[row_group_idx]
        batch_idx = local_index // self.batch_size
        row_idx_in_batch = local_index % self.batch_size
        return row_group_idx, batch_idx, row_idx_in_batch

    def _load_batch_rows(self, row_group_idx, batch_idx):
        cache_key = (row_group_idx, batch_idx)
        if cache_key in self._cache:
            rows = self._cache.pop(cache_key)
            self._cache[cache_key] = rows
            return rows

        self._ensure_open()
        iterator = self._parquet_file.iter_batches(
            row_groups=[row_group_idx],
            columns=self.columns,
            batch_size=self.batch_size,
            use_threads=False,
        )
        try:
            batch = next(islice(iterator, batch_idx, None))
        except StopIteration as exc:
            raise IndexError(
                f"Batch {batch_idx} not found in row group {row_group_idx}."
            ) from exc

        rows = batch.to_pylist()
        self._cache[cache_key] = rows
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return rows

    def __getitem__(self, index):
        row_group_idx, batch_idx, row_idx_in_batch = self._locate_row(index)
        rows = self._load_batch_rows(row_group_idx, batch_idx)
        return rows[row_idx_in_batch]


class VLMConversationDataset(Dataset):
    def __init__(
        self,
        data_path,
        tokenizer,
        max_length=1024,
        image_root=None,
        image_processor_config=None,
        patch_size=14,
        spatial_merge_size=2,
        add_system_ratio=0.2,
        empty_think_ratio=0.2,
    ):
        super().__init__()
        self.data_path = Path(data_path).expanduser().resolve()
        self.data_root = self.data_path.parent
        self.image_root = Path(image_root).expanduser().resolve() if image_root else self.data_root
        self.samples = _load_records(self.data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.add_system_ratio = add_system_ratio
        self.empty_think_ratio = empty_think_ratio

        processor_cfg = self._load_image_processor_config(image_processor_config)
        crop_cfg = processor_cfg.get("crop_size", {})
        size_cfg = processor_cfg.get("size", {})
        self.crop_height = int(crop_cfg.get("height", size_cfg.get("shortest_edge", 560)))
        self.crop_width = int(crop_cfg.get("width", size_cfg.get("shortest_edge", 560)))
        self.resize_shortest_edge = int(size_cfg.get("shortest_edge", self.crop_height))
        self.image_mean = torch.tensor(processor_cfg.get("image_mean", [0.48145466, 0.4578275, 0.40821073]), dtype=torch.float32).view(3, 1, 1)
        self.image_std = torch.tensor(processor_cfg.get("image_std", [0.26862954, 0.26130258, 0.27577711]), dtype=torch.float32).view(3, 1, 1)
        self.grid_h = self.crop_height // self.patch_size
        self.grid_w = self.crop_width // self.patch_size
        self.merge_unit = self.spatial_merge_size ** 2
        patch_tokens_per_image = self.grid_h * self.grid_w
        if patch_tokens_per_image % self.merge_unit != 0:
            raise ValueError(
                "Image patch token count must be divisible by spatial_merge_size**2. "
                f"Got grid {self.grid_h}x{self.grid_w} and merge unit {self.merge_unit}."
            )
        self.image_tokens_per_image = patch_tokens_per_image // self.merge_unit

        self.vision_bos_token = getattr(tokenizer, "vision_bos_token", "<|vision_start|>")
        self.vision_eos_token = getattr(tokenizer, "vision_eos_token", "<|vision_end|>")
        self.image_token = getattr(tokenizer, "image_token", "<|image_pad|>")
        self.single_image_placeholder = (
            self.vision_bos_token
            + (self.image_token * self.image_tokens_per_image)
            + self.vision_eos_token
        )

        self.bos_id = tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids

    def _load_image_processor_config(self, image_processor_config):
        if image_processor_config is None:
            return {}
        config_path = Path(image_processor_config).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Image processor config not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def __len__(self):
        return len(self.samples)

    def _get_sample(self, index):
        return self.samples[index]

    def _pre_process_chat(self, conversations):
        if not conversations:
            return conversations
        if any(message.get("tools") for message in conversations):
            return conversations
        if conversations[0].get("role") != "system" and random.random() < self.add_system_ratio:
            return [{"role": "system", "content": random.choice(SYSTEM_PROMPTS)}] + conversations
        return conversations

    def _post_process_chat(self, prompt_content):
        empty_think = "<think>\n\n</think>\n\n"
        if empty_think in prompt_content and random.random() > self.empty_think_ratio:
            prompt_content = prompt_content.replace(empty_think, "")
        return prompt_content

    def _parse_conversations(self, sample):
        conversations = sample.get("conversations", sample.get("messages"))
        conversations = _maybe_json_loads(conversations)
        if conversations is None:
            raise KeyError("Expected `conversations` or `messages` in multimodal sample.")

        parsed = []
        for message in conversations:
            msg = dict(message)
            if msg.get("tool_calls") and isinstance(msg["tool_calls"], str):
                msg["tool_calls"] = json.loads(msg["tool_calls"])
            parsed.append(msg)
        return parsed

    def _parse_image_sources(self, sample):
        image_sources = []

        image_bytes = _ensure_list(_maybe_json_loads(sample.get("image_bytes")))
        for item in image_bytes:
            if isinstance(item, (bytes, bytearray, memoryview)):
                image_sources.append(bytes(item))

        if image_sources:
            return image_sources

        image_field = sample.get("images", sample.get("image"))
        for item in _ensure_list(_maybe_json_loads(image_field)):
            if item is not None:
                image_sources.append(item)

        if image_sources:
            return image_sources

        image_names = _ensure_list(_maybe_json_loads(sample.get("image_names")))
        for item in image_names:
            if item is not None:
                image_sources.append(item)

        return image_sources

    def _load_image(self, source):
        if isinstance(source, (bytes, bytearray, memoryview)):
            return Image.open(BytesIO(bytes(source))).convert("RGB")

        if isinstance(source, dict):
            if "bytes" in source:
                return Image.open(BytesIO(bytes(source["bytes"]))).convert("RGB")
            if "path" in source:
                source = source["path"]
            elif "image" in source:
                source = source["image"]

        if not isinstance(source, str):
            raise TypeError(f"Unsupported image source type: {type(source).__name__}")

        image_path = Path(source)
        if not image_path.is_absolute():
            candidate = (self.image_root / image_path).resolve()
            if candidate.exists():
                image_path = candidate
            else:
                image_path = (self.data_root / image_path).resolve()

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        return Image.open(image_path).convert("RGB")

    def _preprocess_image(self, image):
        width, height = image.size
        scale = self.resize_shortest_edge / float(min(width, height))
        resized_width = max(int(round(width * scale)), self.crop_width)
        resized_height = max(int(round(height * scale)), self.crop_height)
        image = image.resize((resized_width, resized_height), resample=Image.BICUBIC)

        left = max((resized_width - self.crop_width) // 2, 0)
        top = max((resized_height - self.crop_height) // 2, 0)
        image = image.crop((left, top, left + self.crop_width, top + self.crop_height))

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        pixel_values = torch.from_numpy(image_array).permute(2, 0, 1).contiguous()
        pixel_values = (pixel_values - self.image_mean) / self.image_std
        return pixel_values

    def _expand_image_placeholders(self, conversations, num_images):
        if num_images == 0:
            return conversations

        expanded = []
        total_placeholders = 0
        used_images = 0

        for message in conversations:
            msg = dict(message)
            content = msg.get("content", "")
            if not isinstance(content, str):
                content = str(content)

            def replace_match(_):
                nonlocal used_images
                used_images += 1
                return self.single_image_placeholder

            content, replaced = IMAGE_TAG_RE.subn(replace_match, content)
            total_placeholders += replaced
            msg["content"] = content
            expanded.append(msg)

        if total_placeholders > num_images:
            raise ValueError(
                f"Sample contains {total_placeholders} image placeholders but only {num_images} images."
            )

        if total_placeholders < num_images:
            missing = "\n".join(
                [self.single_image_placeholder for _ in range(num_images - total_placeholders)]
            )
            user_index = next(
                (idx for idx, message in enumerate(expanded) if message.get("role") == "user"),
                0,
            )
            prefix = expanded[user_index].get("content", "")
            expanded[user_index]["content"] = f"{missing}\n{prefix}" if prefix else missing

        return expanded

    def _create_chat_prompt(self, conversations):
        messages = []
        tools = None
        for message in conversations:
            msg = dict(message)
            if msg.get("role") == "system" and msg.get("tools"):
                tools = json.loads(msg["tools"]) if isinstance(msg["tools"], str) else msg["tools"]
            messages.append(msg)
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools,
        )

    def _generate_labels(self, input_ids):
        labels = [-100] * len(input_ids)
        idx = 0
        while idx < len(input_ids):
            if input_ids[idx : idx + len(self.bos_id)] == self.bos_id:
                start = idx + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for pos in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[pos] = input_ids[pos]
                idx = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                idx += 1
        return labels

    def __getitem__(self, index):
        sample = self._get_sample(index)
        conversations = self._parse_conversations(sample)
        image_sources = self._parse_image_sources(sample)
        conversations = self._pre_process_chat(conversations)
        conversations = self._expand_image_placeholders(conversations, len(image_sources))

        prompt = self._create_chat_prompt(conversations)
        prompt = self._post_process_chat(prompt)

        input_ids = self.tokenizer(prompt).input_ids[: self.max_length]
        attention_mask = [1] * len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        attention_mask += [0] * (self.max_length - len(attention_mask))
        labels = self._generate_labels(input_ids)

        images = []
        for source in image_sources:
            image = self._load_image(source)
            images.append(self._preprocess_image(image))

        if images:
            image_tensor = torch.stack(images, dim=0)
            image_grid_thw = torch.tensor(
                [[1, self.grid_h, self.grid_w] for _ in images],
                dtype=torch.long,
            )
        else:
            image_tensor = torch.empty((0, 3, self.crop_height, self.crop_width), dtype=torch.float32)
            image_grid_thw = torch.empty((0, 3), dtype=torch.long)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "images": image_tensor,
            "image_grid_thw": image_grid_thw,
        }


class VLMPretrainDataset(VLMConversationDataset):
    pass


class VLMSFTDataset(VLMConversationDataset):
    pass


def vlm_collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)
    labels = torch.stack([item["labels"] for item in batch], dim=0)
    attention_mask = torch.stack([item["attention_mask"] for item in batch], dim=0)

    image_tensors = [item["images"] for item in batch if item["images"].numel() > 0]
    image_grids = [item["image_grid_thw"] for item in batch if item["image_grid_thw"].numel() > 0]

    if image_tensors:
        images = torch.cat(image_tensors, dim=0)
        image_grid_thw = torch.cat(image_grids, dim=0)
    else:
        images = torch.empty((0, 3, 0, 0), dtype=torch.float32)
        image_grid_thw = torch.empty((0, 3), dtype=torch.long)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "images": images,
        "image_grid_thw": image_grid_thw,
    }
