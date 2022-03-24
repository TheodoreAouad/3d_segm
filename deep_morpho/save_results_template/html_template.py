def html_template():
    return """
    <!DOCTYPE html>
    <html>
      <link href="/hdd/aouadt/these/projets/3d_segm/deep_morpho/save_results_template/html_template.css" rel="stylesheet">
      <head>
        <title>{title}</title>
      </head>
      <body>
        <h2>Tensorboard paths</h2>
        <p>{tb_paths}</p>
        <h2>Global Args</h2>
        <p>{global_args}</p>
        <h2>Changing args</h2>
        <p>{changing_args}</p>
        <h2>Table</h2>
        <p>{table}</p>
        <h2>Results</h2>
        <span>{results}</span>
      </body>
    </html>
    """
