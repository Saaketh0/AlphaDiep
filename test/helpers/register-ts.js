const fs = require('node:fs');
const path = require('node:path');
const ts = require(path.join(__dirname, '../../ts-server/node_modules/typescript'));

require.extensions['.ts'] = function compileTypeScript(module, filename) {
  const source = fs.readFileSync(filename, 'utf8');
  const { outputText } = ts.transpileModule(source, {
    compilerOptions: {
      esModuleInterop: true,
      module: ts.ModuleKind.CommonJS,
      target: ts.ScriptTarget.ES2021,
      resolveJsonModule: true
    },
    fileName: filename
  });
  module._compile(outputText, filename);
};
