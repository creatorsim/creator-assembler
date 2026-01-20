import init, * as wasm from "../pkg/web/creator_assembler.js";
import * as assembler from "./assembler.mjs";

await init({})

const json_arch = await (await fetch("../tests/architecture.json")).text()

const arch = assembler.load(wasm, json_arch)
console.log(arch.toString())
window["arch"] = arch

const src = document.getElementById("src");
const out = document.getElementById("result");

document.getElementById("compile_btn").onclick = function () {
  try {
    const compiled = assembler.compile(wasm, arch, src.value);
    window["instructions"] = compiled.instructions
    window["data"] = compiled.data
    window["instructions"] = compiled.label_table
    console.log(compiled.instructions)
    console.log(compiled.data)
    console.log(compiled.label_table)
    out.innerHTML = compiled.msg;
  } catch (e) {
    out.innerHTML = e;
  }
}
