"""Exercise the frontend parameter rules against the live Python request schema."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from webui.services import moss_service, moss_workspace, qwen21_workspace

ROOT = Path(__file__).resolve().parents[1]


def test_live_moss_schema_groups_and_action_payloads():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the frontend/backend contract")
    script = r"""
import fs from 'node:fs';
import assert from 'node:assert/strict';
import {initialDraft, checkedValues} from './webui/frontend/src/utils/qwen21.ts';
import {groupModelConfigFields, modelSubBlocks, modelFieldActive, modelActionValues} from './webui/frontend/src/utils/modelConfig.ts';
import {modelInputLabel, modelFieldReference} from './webui/frontend/src/i18n/modelWorkspace.ts';
const schema = JSON.parse(fs.readFileSync(0, 'utf8'));
for (const phase of ['cache', 'train', 'generate', 'tune']) for (const field of schema[phase]) {
  const reference = modelFieldReference('moss', 'cn', phase, field, '');
  assert.equal(reference.original, field.help, `${phase}.${field.name} lost its original documentation`);
  assert.match(reference.translation, /[\u4e00-\u9fff]/);
  assert.doesNotMatch(reference.translation, /暂无中文翻译/, `${phase}.${field.name} has no translation`);
  for (const language of ['en', 'cn', 'ja', 'ko']) assert.equal(modelInputLabel('moss', language, field.name, '', phase), field.name);
}

const outputs = [];
for (const algo of ['lora', 'lokr', 'loha']) {
  const draft = initialDraft(schema);
  Object.assign(draft.train, {algo, svd_slice: 1, svd_slice_count: 4, seed: 11, rank: 16, precision: 'bf16'});
  Object.assign(draft.generate, {prompt: 'test', seed: 22, precision: 'fp16'});
  Object.assign(draft.tune, {seed: 33, rank: 64, precision: 'auto'});
  const before = JSON.stringify(draft);
  const groups = groupModelConfigFields(schema, draft, 'moss');
  const seen = new Set();
  for (const group of groups) for (const section of group.sections) {
    const blocks = modelSubBlocks(group.id, section.fields);
    assert.equal(blocks.flatMap(b => b.fields).length, section.fields.length);
    for (const field of blocks.flatMap(b => b.fields)) {
      const key = `${section.section}.${field.name}`;
      assert.ok(!seen.has(key), `duplicate ${key}`);
      seen.add(key);
    }
  }
  for (const phase of ['train', 'generate', 'tune']) {
    for (const field of schema[phase]) {
      if (field.name === 'algo' || schema.model_field_names.includes(field.name)) continue;
      assert.equal(seen.has(`${phase}.${field.name}`), modelFieldActive('moss', phase, field.name, draft[phase]));
    }
    const values = checkedValues(schema[phase], modelActionValues('moss', phase, schema[phase], {...draft[phase], ...draft.models}));
    outputs.push({algo, phase, values});
  }
  assert.equal(JSON.stringify(draft), before);
}
console.log(JSON.stringify(outputs));
"""
    result = subprocess.run(
        [node, "--experimental-strip-types", "--input-type=module", "-e", script],
        input=moss_service.read_schema(moss_workspace.MODEL_FIELDS).model_dump_json(),
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=30,
        check=True,
    )
    for item in json.loads(result.stdout):
        phase = item["phase"]
        cls = moss_service.REQUEST_TYPES[f"moss-{phase}"]
        req = moss_service.request_from_values(cls, item["values"])
        assert req.seed == {"train": 11, "generate": 22, "tune": 33}[phase]
        assert (
            req.precision
            == {"train": "bf16", "generate": "fp16", "tune": "auto"}[phase]
        )
        if phase == "train":
            assert req.rank == 16
            assert req.svd_slice == (1 if item["algo"] == "lora" else 0)
        if phase == "tune":
            assert req.rank == 64


def test_profiles_round_trip_without_crossing_phases_or_families(tmp_path, monkeypatch):
    for workspace in (moss_workspace, qwen21_workspace):
        monkeypatch.setattr(
            workspace, "resolve_under_home", lambda value: tmp_path / value
        )
    for name, offset in (("profile_a", 0), ("profile_b", 100)):
        profile = moss_workspace.Profile(
            models={"model_dir": f"/models/{name}"},
            cache={
                "out": f"post_image_dataset/moss/{name}",
                "vae": f"/{name}-cache.safetensors",
                "text_encoder": "int8",
            },
            train={
                "rank": 16 + offset,
                "seed": 11 + offset,
                "precision": "bf16",
                "output": f"output/ckpt_moss/{name}.safetensors",
            },
            generate={
                "seed": 22 + offset,
                "precision": "fp16",
                "vae": f"/{name}-generate.safetensors",
                "text_encoder": "fp8",
            },
            tune={
                "rank": 64 + offset,
                "seed": 33 + offset,
                "precision": "auto",
                "output": f"output/ckpt_moss/{name}.json",
            },
        )
        normalized = moss_workspace.save_profile(name, profile)
        assert moss_workspace.read_profile(name) == normalized
    qwen21_workspace.save_profile(
        "profile_a",
        qwen21_workspace.Profile(models={}, cache={}, train={"rank": 999}, generate={}),
    )
    a = moss_workspace.read_profile("profile_a")
    b = moss_workspace.read_profile("profile_b")
    assert a.train["rank"] == 16 and b.train["rank"] == 116
    assert a.tune["rank"] == 64 and b.tune["rank"] == 164
    assert a.generate["seed"] == 22 and b.generate["seed"] == 122
    assert a.cache["vae"] != a.generate["vae"]
    assert a.cache["text_encoder"] == "int8" and a.generate["text_encoder"] == "fp8"
    assert a.train["output"].endswith(".safetensors") and a.tune["output"].endswith(
        ".json"
    )
