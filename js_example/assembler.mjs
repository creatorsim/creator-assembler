/**
 * @typedef {Object} CompilationResult
 * @property {Object[]} instructions
 * @property {Object[]} data
 * @property {string} msg
 **/

/**
 * @param {import("../pkg/web/creator_assembler.d.ts")} wasm
 * @param {string} json_arch
 * @returns {import("../pkg/web/creator_assembler.d.ts").ArchitectureJS}
 **/
export function load(wasm, json_arch) {
    const arch = wasm.ArchitectureJS.from_json(json_arch);
    return arch
}

/**
 * @param {import("../pkg/web/creator_assembler.d.ts")} wasm
 * @param {import("../pkg/web/creator_assembler.d.ts").DataCategoryJS} category
 * @returns {string}
 **/
function data_category(wasm, category) {
    switch (category) {
        case wasm.DataCategoryJS.Number: return "Number";
        case wasm.DataCategoryJS.String: return "String";
        case wasm.DataCategoryJS.Space: return "Space";
        case wasm.DataCategoryJS.Padding: return "Padding";
    }
}

/**
 * @param {import("../pkg/web/creator_assembler.d.ts")} wasm
 * @param {import("../pkg/web/creator_assembler.d.ts").ArchitectureJS} arch
 * @param {string} code
 * @returns {CompilationResult}
 **/
export function compile(wasm, arch, code) {
    const color = typeof document != "undefined"? wasm.Color.Html : wasm.Color.Ansi;
    const compiled = arch.compile(code, 0, "{}", false, color);
    const result = {
        instructions: compiled.instructions.map(x => ({
            address: x.address,
            labels: x.labels,
            loaded: x.loaded,
            binary: x.binary,
            user: x.user
        })),
        data: compiled.data.map(data => ({
            labels: data.labels(),
            addr: data.address(),
            size: data.size(),
            type: data.type(),
            value: data.value(false),
            value_human: data.value(true),
            category: data_category(wasm, data.data_category()),
        })),
        label_table: compiled.label_table.reduce(
            (tbl, x) => {
                tbl[x.name] = { address: x.address, global: x.global };
                return tbl
            },
            {},
        ),
        msg: `Code compiled successfully\n` + compiled.toString(),
    };
    return result;
}
