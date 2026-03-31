import resource

from flask import Flask

from trainer import db
from trainer.routes import annotate, api, models, projects


def _raise_fd_limit(target: int = 4096) -> None:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    new_soft = min(target, hard)
    if new_soft > soft:
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))


def create_app() -> Flask:
    _raise_fd_limit()
    app = Flask(__name__)
    db.init_db()
    db.fail_stale_training_runs()
    db.fail_stale_quality_training_runs()
    app.register_blueprint(projects.bp)
    app.register_blueprint(annotate.bp)
    app.register_blueprint(api.bp)
    app.register_blueprint(models.bp)
    return app


app = create_app()








