 ========================= Control Path =========================

  EngineCore.step()
    -> MultiprocExecutor.collective_rpc(method, args, ...)
       -> rpc_broadcast_mq.enqueue((method,args,kwargs,output_rank))
          -> 每个 WorkerProc.worker_busy_loop() dequeue()
             -> 执行 getattr(worker, method)(*args, **kwargs)
             -> 把执行结果写回 worker_response_mq（或指定 output_rank）

  主进程侧:
    collective_rpc.get_response()
      <- 从 response_mqs dequeue()
      <- FutureWrapper 按“请求发送顺序”消费响应，避免串读

  ===============================================================

  ========================== Data Flow ===========================

  (1) 请求进入
  Client request
    -> EngineCoreClient
    -> EngineCoreProc / EngineCore.input_queue
    -> Scheduler 产出 SchedulerOutput

  (2) 调模型
  SchedulerOutput
    -> 作为 execute_model RPC 参数，经 rpc_broadcast_mq 发给 workers

  (3) worker 计算
  Workers 执行 forward/sample
    -> 生成 ModelRunnerOutput

  (4) 回传
  ModelRunnerOutput
    -> worker_response_mq -> MultiprocExecutor
    -> (可选) KVOutputAggregator 聚合多 worker 的 kv_connector_output
    -> 返回 EngineCore

  (5) 出口
  EngineCore 组装 EngineCoreOutputs
    -> EngineCoreProc.output_queue / ZMQ
    -> EngineCoreClient
    -> Detokenizer / Response builder
    -> 返回用户

  ===============================================================