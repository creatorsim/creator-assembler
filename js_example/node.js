const fs = require("fs");
const wasm = require('../pkg/nodejs/creator_assembler.js');
/**@type {import("./assembler.mjs")} assembler */
import("./assembler.mjs").then(assembler => {
    const json_arch = fs.readFileSync(__dirname + "/../tests/architecture.json", "utf8")

    const arch = assembler.load(wasm, json_arch)
    const src = fs.readFileSync(process.argv[2], "utf8")

    try {
        const compiled = assembler.compile(wasm, arch, src);
        console.log(compiled.msg);
        console.log(compiled.instructions);
        console.log(compiled.data);
        console.log(compiled.label_table);
    } catch (e) {
        console.error(e);
    }
});

