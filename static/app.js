// ── WebGL animated footer canvas ─────────────────────────────────────────────
(function () {
  const canvas = document.getElementById("canvas");
  if (!canvas) return;
  const gl = canvas.getContext("webgl");
  if (!gl) return;

  const vs = document.getElementById("vertexShader");
  const fs = document.getElementById("fragmentShader");
  if (!vs || !fs) return;

  const programInfo = twgl.createProgramInfo(gl, [vs.textContent, fs.textContent]);
  const arrays = { position: [-1,-1,0, 1,-1,0, -1,1,0, -1,1,0, 1,-1,0, 1,1,0] };
  const bufferInfo = twgl.createBufferInfoFromArrays(gl, arrays);

  function render(time) {
    twgl.resizeCanvasToDisplaySize(gl.canvas, 0.5);
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    const uniforms = {
      u_time: time * 0.002,
      u_resolution: [gl.canvas.width, gl.canvas.height],
    };
    gl.useProgram(programInfo.program);
    twgl.setBuffersAndAttributes(gl, programInfo, bufferInfo);
    twgl.setUniforms(programInfo, uniforms);
    twgl.drawBufferInfo(gl, bufferInfo);
    requestAnimationFrame(render);
  }
  requestAnimationFrame(render);
})();
