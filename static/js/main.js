window.onload = function(){
        var key_log = [];
        var tmp_key = {};


        var start_t = null;
        var start_prev_t = null;
        $('#main_input').on("keydown", function(event){
            if(!start_t){ start_t = $.now(); }
            tmp_key.vk = event.key;

            let end_t = $.now();
            var time_elapsed = end_t - (start_prev_t == null ? end_t : start_prev_t);
            tmp_key.diff_prev_t = time_elapsed;
        }).on("keyup", function(){
            let end_t = $.now();
            var time_elapsed = end_t - (start_t == null ? end_t : start_t);

            tmp_key.start_t = start_t;
            tmp_key.end_t = end_t;
            tmp_key.diff_t = time_elapsed;
            key_log.push(tmp_key);

            start_prev_t = start_t;
            start_t = null;

            console.log(["Key: ", tmp_key.vk, ", time prev: ", tmp_key.diff_prev_t, ", time: ", time_elapsed, "ms"].join(""));
        });
    };
